# Teacher qualification evaluation

This directory prepares and runs the English four-source comparison between the sealed SFT-VH merged student and `Qwen/Qwen3-8B`. It does not train OPD.

## Frozen contract

- Prompt: sealed SFT-VH V1 prompt `qmd-student-expansion-v2-vh-v1`.
- Output: exactly one `hyde:` followed by one or two `vec:` lines.
- Generation: greedy, BF16, no quantization, one response/query, 256 completion tokens, model length 1024.
- Retrieval: complete corpus, `rerank=false`.
- Any invalid output from either model produces `not_evaluated`.

## Local preparation

Static data check (does not copy corpora):

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.prepare_qualification_data \
  --config finetune/experiments/opd/config/teacher-qualification-v1.yaml \
  --check
```

Local synthetic dry-run (uses a temporary directory and deletes it on exit):

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.qualify_teacher --dry-run
```

To inspect artifacts, pass an output outside the repository, for example:

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.qualify_teacher \
  --dry-run --output /tmp/qmd-teacher-qualification-dry-run
```

Prepare a portable input tree in a temporary/staging directory. The tree uses corpus symlinks to avoid duplicate local storage:

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.prepare_qualification_data \
  --config finetune/experiments/opd/config/teacher-qualification-v1.yaml \
  --output /tmp/qmd-teacher-qualification-input
```

When archiving, dereference those symlinks so AutoDL receives the complete NFCorpus, FiQA, FreshStack, and SciFact corpora:

```bash
tar -chzf /tmp/qmd-teacher-qualification-input.tar.gz \
  -C /tmp qmd-teacher-qualification-input
shasum -a 256 /tmp/qmd-teacher-qualification-input.tar.gz
```

Transfer these items to AutoDL:

- this repository revision;
- the prepared input archive;
- `artifacts/sft-runs/02-public-main-v2-vh-prompt-v1/training/public-main-v2-vh-prompt-v1/final-adapter/`;
- the sealed SFT-VH release manifest/data referenced by the config.

Do not transfer merged weights; merge them on AutoDL.

## AutoDL execution

Resolve `Qwen/Qwen3-8B` to an immutable Hugging Face commit before the formal run:

```bash
export QMD_TEACHER_REVISION=<40-character-commit-sha>
```

Check the local identities without loading a model:

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.merge_v2_adapter \
  --config finetune/experiments/opd/config/teacher-qualification-v1.yaml \
  --check
```

Download the pinned base/teacher revisions and embedding model into the AutoDL cache, then merge the student on CUDA:

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.merge_v2_adapter \
  --config finetune/experiments/opd/config/teacher-qualification-v1.yaml \
  --output /root/autodl-tmp/qmd/models/qmd-qwen3-1.7b-v2-merged
```

Create and vectorize each QMD collection declared by the prepared benchmark profiles, then run `qmd bench <benchmark-dir> --check-index` for all four sources. The evaluator refuses missing/mismatched indexes.

Formal comparison:

```bash
PYTHONPATH=finetune finetune/.venv/bin/python -m experiments.opd.qualify_teacher \
  --config finetune/experiments/opd/config/teacher-qualification-v1.yaml \
  --bundle /root/autodl-tmp/qmd/teacher-qualification-input \
  --merged-model /root/autodl-tmp/qmd/models/qmd-qwen3-1.7b-v2-merged \
  --output /root/autodl-tmp/qmd/runs/teacher-qualification-<run-id>
```

The formal output ends in `go`, `no_go`, or `not_evaluated`. A local dry-run can only produce `not_evaluated`.

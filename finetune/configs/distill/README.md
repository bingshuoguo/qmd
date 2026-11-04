# Distillation configuration

Run these commands from the repository root. Prompt file paths inside every
profile are relative to that root, not to the environment file's directory.

## Current configurations

| Root file | Purpose |
| --- | --- |
| `.env.distill.deepseek-json4-v2.local` | Local settings required by `public-distill:generate`. |
| `.env.distill.deepseek-json4-v2.example` | Template for the frozen public-distillation profile. |

The `.local` file contains private settings and is ignored by Git. Copy the
template to `.local` and fill in your credentials; never commit the key.

Historical prompt and environment profiles from the SciFact teacher-selection
pilot are archived on git tag `archive/completion-only-sft-20261104`
(`finetune/benchmarks/distill/prompts/`, `finetune/configs/distill/legacy/`).
Recover one with `git checkout <tag> -- <path>`; they are not part of the
main-line tree.

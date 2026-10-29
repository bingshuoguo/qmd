# Distillation configuration

Run these commands from the repository root. Prompt file paths inside every
profile are relative to that root, not to the environment file's directory.

## Current configurations

| Root file | Purpose |
| --- | --- |
| `.env.distill.local` | Local settings loaded by `distill:generate` when present. |
| `.env.distill.deepseek-json4-v2.local` | Local settings required by `public-distill:generate`. |
| `.env.distill.deepseek-json4-v2.example` | Template for the frozen public-distillation profile. |
| `.env.distill.openai-v2.example` | Template for the OpenAI-compatible Responses API profile. |

The `.local` files contain private settings and are ignored by Git. Moving
historical profiles does not change these active settings or either command.

## Historical profiles

`legacy/` retains the original template bytes for experiment reproduction:

| File | Profile |
| --- | --- |
| `.env.distill.deepseek-json4-repro.example` | Original four-candidate JSON profile. |
| `.env.distill.deepseek-v1.example` | DeepSeek V1 semantic-safe profile. |
| `.env.distill.deepseek-v2.example` | General-purpose DeepSeek V2 profile. |
| `.env.distill.deepseek-v3.example` | SciFact-specific DeepSeek V3 profile. |
| `.env.distill.v1.example` | Original inline V1 prompt. |
| `.env.distill.v2.example` | Original inline V2 prompt. |

For an explicit one-off run with a historical profile, supply its environment
file directly and provide credentials through the process environment:

```sh
node --env-file=finetune/configs/distill/legacy/.env.distill.deepseek-v2.example \
  --import tsx finetune/benchmarks/distill/generate.ts --help
```

To select a historical profile for `distill:generate`, back up your current
`.env.distill.local`, copy the desired template there, and fill in your local
credentials. This replaces the active profile; copying is not required for
reading or scoring existing experiment artifacts.

## Spare local configurations

`local/` is ignored as a whole. The existing spare JSON4 repro and OpenAI V2
`.local` files were moved there without changing their contents. Use one
directly with `node --env-file=finetune/configs/distill/local/<filename> ...`,
or copy it to `.env.distill.local` after backing up the active configuration.

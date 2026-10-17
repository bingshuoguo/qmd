# Frozen public evaluation

Use `run_8b_sft_eval.py` for the seven-benchmark Qwen3-8B SFT evaluation.
It runs only the **all** retrieval arm. Generation stays greedy with a
768-token budget, and the existing frozen index and adapter are reused.

## Preflight from a clean code worktree

Use the existing evaluation interpreter that matches `finetune/requirements-eval.txt`.
The runner checks installed versions and records the executable and versions;
it never installs or upgrades packages. Keep training environments separate.

```sh
/root/miniconda3/bin/python -B finetune/benchmarks/public-eval/run_8b_sft_eval.py \
  --asset-root /root/autodl-tmp/qmd --dry-run
```

`--asset-root` locates existing datasets, release, index and configuration, while
scripts run from the current code worktree. Override `--model`, `--adapter`,
`--release`, `--index-path` or `--config-dir` when those assets move. The model
revision must be an immutable commit; a local model path must already contain
that revision's files (the revision argument does not verify local weights).

Preflight validates the sealed release and its stored prompts, the full V2-VH
prompt hash, frozen inputs, index fingerprints and output collisions. Index
inspection opens SQLite read-only without migrations. Exact normalized query
overlap is reported; this is not a complete semantic leakage audit.

When GPU capacity is available, replace `--dry-run` with `--run` and choose a
fresh `--variant`. The default `8b-sft-v2vh-sealed-v1` separates the corrected
prompt from earlier evaluations. Preflight provenance is persisted under
`finetune/evaluation-runs/`; raw generation manifests record the template,
release hashes, source hash and runtime environment. Never reuse a partial run
ID or overwrite old results to resume it.

## Prompt and result contracts

- SFT generation uses `--release-manifest`, sharing the trainer's release loader
  and exact prompt renderer. The template is not copied into the runner.
- `--prompt-version v1`, `v2a-pareto` and `v2-vh-legacy` are explicit historical
  alternatives. The ambiguous old selector `v2-vh` is rejected. The abbreviated
  legacy template has a distinct version name for new runs; historical manifests
  are immutable.
- Raw output and stopping metadata are evidence. Do not trim output, relabel a
  length stop as EOS, or replace raw hashes after post-processing. Derived
  diagnostics must use separate files and record their source hash and policy.
- Keep model weights, indexes, logs, snapshots, one-off transfer scripts and
  generated reports outside code commits. Preserve their paths and provenance
  in separate operational inventories.
- OPD qualification is a separate experiment. Its base teacher, dataset contract
  and diagnostic policies are not changed by this runner.

## CPU regression checks

```sh
cd finetune
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -m unittest discover -s tests -p 'test_evaluation_contract.py'
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -m unittest discover -s tests -p 'test_generate_expansions.py'
```

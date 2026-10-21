# Sealed-release expansion evaluation

Generate query expansions from a sealed SFT release, then score them with
`qmd bench`. Three steps, each checking the previous step's output:

```sh
# 1. Raw generation (GPU recommended), in an environment from finetune/requirements-eval.txt
python finetune/benchmarks/public-eval/generate_expansions.py \
  --release-manifest <release>/experiments/<id>/release-manifest.json \
  --benchmark <benchmark-dir> --variant sft --model <base-snapshot> \
  --adapter <run>/final-adapter
# Omit --adapter (and use --variant base) for the base arm; the prompt is identical.
# The base arm is not a quality baseline; see the end of this file.

# 2. Validate raw output and write the canonical expansions/<variant>.jsonl
node --import tsx finetune/benchmarks/public-eval/materialize-expansions.ts \
  --benchmark <benchmark-dir> --variant sft

# 3. Retrieval. Freeze the index once per benchmark, then score.
#    --run reads expansions/<variant>.jsonl; --model is recorded as the run's label.
qmd bench <benchmark-dir> --check-index
qmd bench <benchmark-dir> --run raw
qmd bench <benchmark-dir> --run sft --model <adapter-label>
```

- The prompt always comes from the release manifest. It is checked against
  every stored training prompt before any model loads.
- `raw-generations/<variant>-manifest.json` records the release, model, adapter
  hash, generation settings, `qmd_commit`/`qmd_dirty`, the `queries.jsonl` hash
  and the raw file hash. Existing raw outputs are never overwritten.
- Materialization refuses a raw file whose hash, row count, or qid/query order
  no longer matches its manifest and the benchmark. An interrupted run resumes
  only for the same raw file.
- A row that hit the token budget or stopped without EOS is recorded as
  `generation_error`. It is never parsed as a successful expansion.

## The base arm is not a quality baseline

Without an adapter, Qwen3 starts every answer with an empty `<think>\n\n</think>`
block. Output is parsed as-is, with no special case for this block, so the
base arm comes out almost entirely `format_error`. Its retrieval numbers are
close to the no-expansion path. Use it to show that the model learned the
output format, not as a measure of expansion quality: compare SFT against
`--run raw` instead.

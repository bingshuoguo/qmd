#!/usr/bin/env node
/**
 * Turn raw model generations into the canonical expansion artifact (spec section 16).
 *
 * The Python generator only produces text.  Parsing happens here, through the
 * same `generateExpansionArtifact` path every other benchmark run uses, so the
 * raw, base and SFT arms are parsed and accounted for by one implementation.
 *
 * Usage:
 *   npx tsx finetune/benchmarks/public-eval/materialize-expansions.ts \
 *     --benchmark <benchmark-dir> --variant base
 */

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import { generateExpansionArtifact } from "../../../src/bench/expansions.js";

const { values } = parseArgs({
  options: {
    benchmark: { type: "string" },
    variant: { type: "string" },
    force: { type: "boolean", default: false },
    help: { type: "boolean", default: false },
  },
});

if (values.help || !values.benchmark || !values.variant) {
  process.stderr.write(
    "Usage: materialize-expansions.ts --benchmark <dir> --variant <name> [--force]\n",
  );
  process.exit(values.help ? 0 : 1);
}

const benchmarkDir = resolve(values.benchmark);
const rawPath = join(benchmarkDir, "raw-generations", `${values.variant}.jsonl`);
if (!existsSync(rawPath)) {
  throw new Error(`Raw generations are missing, run generate_expansions.py first: ${rawPath}`);
}

/**
 * Raw generations are keyed by query text because that is what the artifact
 * generator hands to the callback. Duplicate query text within one benchmark
 * would make the lookup ambiguous, so reject it rather than guess.
 */
const rawByQuery = new Map<string, string>();
for (const line of readFileSync(rawPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const record = JSON.parse(line) as { query?: unknown; raw_output?: unknown };
  if (typeof record.query !== "string" || typeof record.raw_output !== "string") {
    throw new Error(`${rawPath}: each row needs string query/raw_output`);
  }
  if (rawByQuery.has(record.query)) {
    throw new Error(`${rawPath}: duplicate query text, cannot map raw output unambiguously`);
  }
  rawByQuery.set(record.query, record.raw_output);
}

const summary = await generateExpansionArtifact({
  benchmarkDir,
  runName: values.variant,
  force: values.force,
  generateRaw: async (query: string) => {
    const raw = rawByQuery.get(query);
    if (raw === undefined) {
      throw new Error(`No raw generation for query: ${JSON.stringify(query)}`);
    }
    return raw;
  },
});

process.stdout.write(`${JSON.stringify({ variant: values.variant, ...summary }, null, 2)}\n`);

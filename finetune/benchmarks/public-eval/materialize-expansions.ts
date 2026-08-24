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
 * generator hands to the callback. Some BEIR splits contain distinct qids with
 * identical text, so each text maps to an input-order queue.
 */
const rawByQuery = new Map<string, Array<{
  rawOutput: string;
  generationError: string | null;
  truncated: boolean;
}>>();
for (const line of readFileSync(rawPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const record = JSON.parse(line) as {
    query?: unknown;
    raw_output?: unknown;
    generation_error?: unknown;
    truncated?: unknown;
  };
  if (typeof record.query !== "string" || typeof record.raw_output !== "string") {
    throw new Error(`${rawPath}: each row needs string query/raw_output`);
  }
  if (record.generation_error !== undefined && record.generation_error !== null
    && typeof record.generation_error !== "string") {
    throw new Error(`${rawPath}: generation_error must be a string or null`);
  }
  if (record.truncated !== undefined && typeof record.truncated !== "boolean") {
    throw new Error(`${rawPath}: truncated must be a boolean`);
  }
  const queue = rawByQuery.get(record.query) ?? [];
  queue.push({
    rawOutput: record.raw_output,
    generationError: record.generation_error ?? null,
    truncated: record.truncated ?? false,
  });
  rawByQuery.set(record.query, queue);
}

const summary = await generateExpansionArtifact({
  benchmarkDir,
  runName: values.variant,
  force: values.force,
  generateRaw: async (query: string) => {
    const queue = rawByQuery.get(query);
    const record = queue?.shift();
    if (record === undefined) {
      throw new Error(`No raw generation for query: ${JSON.stringify(query)}`);
    }
    if (record.generationError !== null) throw new Error(record.generationError);
    if (record.truncated) throw new Error("model output reached max token budget");
    return record.rawOutput;
  },
});

process.stdout.write(`${JSON.stringify({ variant: values.variant, ...summary }, null, 2)}\n`);

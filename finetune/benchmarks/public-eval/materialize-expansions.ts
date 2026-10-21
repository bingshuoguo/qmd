#!/usr/bin/env node
/**
 * Turn raw model generations (generate_expansions.py) into the canonical
 * expansion artifact, parsed by the same code path as every other run.
 *
 * Before writing anything, the raw file is checked against its manifest and
 * the benchmark: file hash, queries.jsonl hash, row count, and per-row
 * qid/query order. Rows are then looked up by qid, so distinct qids with
 * identical query text stay distinct.
 *
 * Usage:
 *   node --import tsx finetune/benchmarks/public-eval/materialize-expansions.ts \
 *     --benchmark <benchmark-dir> --variant sft [--force]
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import { generateExpansionArtifact } from "../../../src/bench/expansions.js";
import { parseQueriesJsonl } from "../../../src/bench/qrels.js";

type RawRow = {
  qid: string;
  query: string;
  raw_output: string;
  truncated: boolean;
  generation_error: string | null;
};

const { values } = parseArgs({
  options: {
    benchmark: { type: "string" },
    variant: { type: "string" },
    force: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help || !values.benchmark || !values.variant) {
  process.stderr.write("Usage: materialize-expansions.ts --benchmark <dir> --variant <name> [--force]\n");
  process.exit(values.help ? 0 : 1);
}

const sha256 = (bytes: Buffer): string => createHash("sha256").update(bytes).digest("hex");

const benchmarkDir = resolve(values.benchmark);
const rawPath = join(benchmarkDir, "raw-generations", `${values.variant}.jsonl`);
const manifestPath = join(benchmarkDir, "raw-generations", `${values.variant}-manifest.json`);
for (const path of [rawPath, manifestPath]) {
  if (!existsSync(path)) throw new Error(`Missing ${path}; run generate_expansions.py first`);
}

const rawBytes = readFileSync(rawPath);
const queriesBytes = readFileSync(join(benchmarkDir, "queries.jsonl"));
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
  variant?: unknown;
  raw_sha256?: unknown;
  queries_sha256?: unknown;
  rows?: unknown;
};
if (manifest.variant !== values.variant) {
  throw new Error(`${manifestPath}: variant ${JSON.stringify(manifest.variant)} != ${values.variant}`);
}
const rawSha256 = sha256(rawBytes);
if (manifest.raw_sha256 !== rawSha256) throw new Error(`${rawPath}: sha256 does not match its manifest`);
if (manifest.queries_sha256 !== sha256(queriesBytes)) {
  throw new Error(`${benchmarkDir}/queries.jsonl changed since the raw generations were produced`);
}

const queries = parseQueriesJsonl(queriesBytes.toString("utf8"));
const rows = rawBytes.toString("utf8").split("\n").filter(line => line.trim()).map((line, index) => {
  const row = JSON.parse(line) as Partial<RawRow>;
  const location = `${rawPath}:${index + 1}`;
  if (typeof row.qid !== "string" || typeof row.query !== "string" || typeof row.raw_output !== "string") {
    throw new Error(`${location}: qid, query and raw_output must be strings`);
  }
  if (typeof row.truncated !== "boolean") throw new Error(`${location}: truncated must be a boolean`);
  if (row.generation_error !== null && typeof row.generation_error !== "string") {
    throw new Error(`${location}: generation_error must be a string or null`);
  }
  return row as RawRow;
});
if (rows.length !== queries.length || manifest.rows !== rows.length) {
  throw new Error(
    `${rawPath}: ${rows.length} rows (manifest ${String(manifest.rows)}) for ${queries.length} queries`,
  );
}
rows.forEach((row, index) => {
  const query = queries[index]!;
  if (row.qid !== query.qid || row.query !== query.query) {
    throw new Error(`${rawPath}:${index + 1}: expected qid ${query.qid}, got ${row.qid} (or query text differs)`);
  }
});
const rowByQid = new Map(rows.map(row => [row.qid, row]));

const summary = await generateExpansionArtifact({
  benchmarkDir,
  runName: values.variant,
  force: values.force,
  identity: { raw_sha256: rawSha256 },
  generateRaw: async (_query, qid) => {
    const row = rowByQid.get(qid)!;
    if (row.generation_error !== null) throw new Error(row.generation_error);
    if (row.truncated) throw new Error("model output reached max token budget");
    return row.raw_output;
  },
});

process.stdout.write(`${JSON.stringify({ variant: values.variant, ...summary }, null, 2)}\n`);

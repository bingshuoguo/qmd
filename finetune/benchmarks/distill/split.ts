#!/usr/bin/env node

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadBenchmarkV2, parseQrelsTsv } from "../../../src/bench/qrels.js";
import {
  SCIFACT_DISTILL_ALGORITHM_VERSION,
  SCIFACT_DISTILL_NORMALIZATION_VERSION,
  SCIFACT_DISTILL_SEED,
  SCIFACT_DISTILL_SPLIT_VERSION,
  SCIFACT_DISTILL_VALIDATION_COUNT,
  buildSciFactDistillMembership,
  parseSciFactSourceQueries,
  sha256,
  type SciFactDistillSplit,
} from "../lib/distill.js";

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run distill:split -- [--source archive/scifact] [--archive archive/scifact.zip] \\",
    "    [--benchmark finetune/benchmarks/qmd-expansion-scifact-v1] \\",
    "    [--output finetune/data/distillation/scifact-v1/split.json] [--force]",
  ].join("\n");
  (exitCode === 0 ? console.log : console.error)(message);
  process.exit(exitCode);
}

function md5(bytes: Uint8Array): string {
  return createHash("md5").update(bytes).digest("hex");
}

const { values } = parseArgs({
  options: {
    source: { type: "string", default: "archive/scifact" },
    archive: { type: "string", default: "archive/scifact.zip" },
    benchmark: {
      type: "string",
      default: "finetune/benchmarks/qmd-expansion-scifact-v1",
    },
    output: {
      type: "string",
      default: "finetune/data/distillation/scifact-v1/split.json",
    },
    force: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});
if (values.help) usage(0);

const sourceRoot = resolve(values.source);
const archivePath = resolve(values.archive);
const benchmarkRoot = resolve(values.benchmark);
const outputPath = resolve(values.output);
if (!existsSync(archivePath)) throw new Error(`SciFact archive not found: ${archivePath}`);
if (existsSync(outputPath) && !values.force) {
  throw new Error(`Split already exists: ${outputPath}`);
}

const queriesBytes = readFileSync(join(sourceRoot, "queries.jsonl"));
const trainQrelsBytes = readFileSync(join(sourceRoot, "qrels", "train.tsv"));
const testQrelsBytes = readFileSync(join(sourceRoot, "qrels", "test.tsv"));
const benchmarkManifestBytes = readFileSync(join(benchmarkRoot, "benchmark.yaml"));
const archiveBytes = readFileSync(archivePath);
const benchmark = loadBenchmarkV2(benchmarkRoot);
const archiveMd5 = md5(archiveBytes);
if (archiveMd5 !== benchmark.manifest.source.archive_md5) {
  throw new Error(
    `SciFact archive MD5 mismatch: expected ${benchmark.manifest.source.archive_md5}, got ${archiveMd5}`,
  );
}
if (sha256(testQrelsBytes) !== benchmark.manifest.source_qrels_sha256) {
  throw new Error("Downloaded test qrels do not match the frozen test benchmark");
}

const membership = buildSciFactDistillMembership(
  parseSciFactSourceQueries(queriesBytes.toString("utf8")),
  parseQrelsTsv(trainQrelsBytes.toString("utf8")),
  parseQrelsTsv(testQrelsBytes.toString("utf8")),
);
if (
  membership.counts.source_train_queries !== 809
  || membership.counts.source_test_queries !== 300
  || membership.counts.excluded_queries !== 4
  || membership.counts.train_queries !== 644
  || membership.counts.val_queries !== 161
) {
  throw new Error(`Unexpected SciFact split counts: ${JSON.stringify(membership.counts)}`);
}

const split: SciFactDistillSplit = {
  version: SCIFACT_DISTILL_SPLIT_VERSION,
  dataset: "scifact-v1",
  source: {
    archive_md5: archiveMd5,
    queries_sha256: sha256(queriesBytes),
    train_qrels_sha256: sha256(trainQrelsBytes),
    test_qrels_sha256: sha256(testQrelsBytes),
  },
  test_benchmark: {
    benchmark_id: benchmark.manifest.benchmark_id,
    benchmark_manifest_sha256: sha256(benchmarkManifestBytes),
  },
  algorithm: {
    version: SCIFACT_DISTILL_ALGORITHM_VERSION,
    normalization: SCIFACT_DISTILL_NORMALIZATION_VERSION,
    seed: SCIFACT_DISTILL_SEED,
    validation_count: SCIFACT_DISTILL_VALIDATION_COUNT,
  },
  ...membership,
};

mkdirSync(dirname(outputPath), { recursive: true });
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
writeFileSync(temporaryPath, `${JSON.stringify(split, null, 2)}\n`, "utf8");
if (values.force) rmSync(outputPath, { force: true });
renameSync(temporaryPath, outputPath);
console.log(JSON.stringify({ output: outputPath, sha256: sha256(readFileSync(outputPath)), ...split.counts }, null, 2));

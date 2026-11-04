#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { inspectBenchmarkIndex } from "../../../src/bench/bench.js";
import { loadBenchmarkV2, loadRetrievalProfile } from "../../../src/bench/qrels.js";
import { createStore } from "../../../src/store.js";

const { values } = parseArgs({
  options: {
    benchmark: { type: "string", multiple: true },
    db: { type: "string" },
  },
  strict: true,
});

if (!values.db || !values.benchmark?.length) {
  throw new Error("Usage: verify_frozen_indexes.ts --db <index.sqlite> --benchmark <dir> [...]");
}

const dbPath = resolve(values.db);
if (!existsSync(dbPath)) throw new Error(`Evaluation index does not exist: ${dbPath}`);

const store = createStore(dbPath);
const verified: Record<string, unknown>[] = [];
try {
  for (const value of values.benchmark) {
    const benchmarkDir = resolve(value);
    const manifestPath = resolve(benchmarkDir, "index-manifest.json");
    if (!existsSync(manifestPath)) {
      throw new Error(`Missing frozen index manifest: ${manifestPath}`);
    }
    const benchmark = loadBenchmarkV2(benchmarkDir);
    const profile = loadRetrievalProfile(benchmarkDir, benchmark.manifest.cutoffs);
    if (profile.collection_name !== benchmark.manifest.benchmark_id) {
      throw new Error(`${benchmark.manifest.benchmark_id}: collection_name must equal benchmark_id`);
    }
    const frozen = JSON.parse(readFileSync(manifestPath, "utf8"));
    const actual = inspectBenchmarkIndex(store, benchmarkDir, benchmark, profile);
    if (JSON.stringify(actual) !== JSON.stringify(frozen)) {
      throw new Error(`${benchmark.manifest.benchmark_id}: current index differs from the frozen manifest`);
    }
    if (actual.pending_embedding_count !== 0) {
      throw new Error(`${benchmark.manifest.benchmark_id}: embeddings are incomplete`);
    }
    verified.push({
      benchmark_id: benchmark.manifest.benchmark_id,
      document_count: actual.document_count,
      vector_document_count: actual.vector_document_count,
      pending_embedding_count: actual.pending_embedding_count,
      index_fingerprint: actual.index_fingerprint,
    });
  }
} finally {
  store.close();
}

process.stdout.write(`${JSON.stringify({ status: "passed", db: dbPath, benchmarks: verified }, null, 2)}\n`);

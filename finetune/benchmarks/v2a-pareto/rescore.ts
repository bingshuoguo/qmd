#!/usr/bin/env node
/**
 * V2-A Pareto: Task 2 — Retrieval Rescoring (spec section 7).
 *
 * Reads canonical-candidates.jsonl from Task 1, groups by input_id,
 * runs raw + per-candidate retrieval with rerank=false, and writes
 * per-query raw-retrieval.jsonl and per-candidate candidate-retrieval.jsonl.
 *
 * Usage:
 *   npx tsx finetune/benchmarks/v2a-pareto/rescore.ts
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync, appendFileSync, renameSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { inspectBenchmarkIndex } from "../../../src/bench/bench.js";
import { loadBenchmarkV2, loadRetrievalProfile, parseQrelsTsv } from "../../../src/bench/qrels.js";
import { scoreCanonicalRanking } from "../../../src/bench/score.js";
import { createStore, retrieveForBenchmarkBatch } from "../../../src/store.js";
import { applyLlamaEnvMitigation } from "../lib/cli.js";

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function readJsonl(path: string): any[] {
  return readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
}

const { values } = parseArgs({
  options: {
    "canonical-candidates": { type: "string" },
    "output-dir": { type: "string" },
    db: { type: "string" },
    concurrency: { type: "string", default: "4" },
  },
});

// --- Paths ---
const FINETUNE_ROOT = resolve(import.meta.dirname, "..", "..");
const PUBLIC_ROOT = join(FINETUNE_ROOT, "data", "public-distill-v0");
const PREPARED_ROOT = join(PUBLIC_ROOT, "prepared");
const V2A_ROOT = join(FINETUNE_ROOT, "data", "public-distill-v3-vh-pareto", "experiments", "public-main-v3-vh-pareto-prompt-v1");

const canonicalPath = values["canonical-candidates"]
  ?? join(V2A_ROOT, "1-canonicalization", "canonical-candidates.jsonl");
const outputDir = values["output-dir"]
  ?? join(V2A_ROOT, "2-retrieval-scoring");

const rawPath = join(outputDir, "raw-retrieval.jsonl");
const candidatePath = join(outputDir, "candidate-retrieval.jsonl");
const partialRawPath = `${rawPath}.partial`;
const partialCandidatePath = `${candidatePath}.partial`;
const manifestPath = join(outputDir, "rescoring-manifest.json");

if (existsSync(rawPath) || existsSync(candidatePath)) {
  throw new Error("Retrieval artifacts already exist. Remove them first to re-run.");
}

// --- Load canonical candidates ---
console.error("Loading canonical candidates...");
const canonicalRecords = readJsonl(canonicalPath);
console.error(`  ${canonicalRecords.length} canonical candidates`);

// Group by input_id
const byInputId = new Map<string, any[]>();
for (const rec of canonicalRecords) {
  const group = byInputId.get(rec.input_id) ?? [];
  group.push(rec);
  byInputId.set(rec.input_id, group);
}
console.error(`  ${byInputId.size} unique query groups`);

// --- Load source benchmarks ---
const SOURCE_IDS = ["fiqa-train", "cqadup-programmers", "cqadup-unix", "nfcorpus-train-dev", "scifact-train"];
const SOURCE_DIRS: Record<string, string> = {
  "fiqa-train": "fiqa-train",
  "cqadup-programmers": "cqadup-programmers",
  "cqadup-unix": "cqadup-unix",
  "nfcorpus-train-dev": "qmd-distill-public-v0-nfcorpus-train-dev",
  "scifact-train": "qmd-distill-public-v0-scifact-train",
};

const sources = new Map<string, {
  root: string;
  benchmark: any;
  profile: any;
  relevant: Map<string, Set<string>>;
  docIdByPath: Map<string, string>;
}>();

for (const sourceId of SOURCE_IDS) {
  const dirName = SOURCE_DIRS[sourceId]!;
  const root = join(PREPARED_ROOT, dirName);
  if (!existsSync(root)) throw new Error(`Source directory not found: ${root}`);
  const benchmark = loadBenchmarkV2(root);
  const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
  // Override: V2-A requires rerank=false
  profile.rerank = false;
  const qrels = parseQrelsTsv(readFileSync(join(root, "qrels.tsv"), "utf8"));
  const relevant = new Map<string, Set<string>>();
  for (const qrel of qrels) {
    if (qrel.relevance !== 1) continue;
    const docs = relevant.get(qrel.qid) ?? new Set<string>();
    docs.add(qrel.doc_id);
    relevant.set(qrel.qid, docs);
  }
  const docIdByPath = new Map(benchmark.documents.map((d: any) => [d.path, d.doc_id]));
  sources.set(sourceId, { root, benchmark, profile, relevant, docIdByPath });
  console.error(`  ${sourceId}: ${relevant.size} queries with qrels, ${benchmark.documents.length} documents`);
}

// --- Initialize store and LLM ---
const dbPath = resolve(values.db ?? join(process.env.HOME!, ".cache", "qmd", "index.sqlite"));
if (!existsSync(dbPath)) throw new Error(`Index does not exist: ${dbPath}`);
applyLlamaEnvMitigation();
const { LlamaCpp } = await import("../../../src/llm.js");
const firstProfile = sources.get(SOURCE_IDS[0]!)!.profile;
const store = createStore(dbPath);
const llm = new LlamaCpp({
  embedModel: firstProfile.embedding_model,
  // No reranker needed for V2-A (rerank=false)
  inactivityTimeoutMs: 5 * 60 * 1000,
  disposeModelsOnInactivity: true,
});
store.llm = llm;

// --- Index manifest check (relaxed for V2-A: embeddings may be incomplete) ---
const indexManifests: Record<string, unknown> = {};
for (const [sourceId, source] of sources) {
  const frozen = JSON.parse(readFileSync(join(source.root, "index-manifest.json"), "utf8"));
  const actual = inspectBenchmarkIndex(store, source.root, source.benchmark, source.profile);
  // V2-A: allow incomplete embeddings (we only warn)
  if (actual.pending_embedding_count !== 0) {
    console.error(`  WARNING: ${sourceId}: ${actual.pending_embedding_count} documents need embeddings`);
  }
  if (actual.document_count !== frozen.document_count) {
    console.error(`  WARNING: ${sourceId}: document count mismatch (frozen=${frozen.document_count}, actual=${actual.document_count})`);
  }
  indexManifests[sourceId] = actual;
}

// --- Retrieve function ---
// --- Initialize partial files ---
if (!existsSync(partialRawPath)) writeFileSync(partialRawPath, "", "utf8");
if (!existsSync(partialCandidatePath)) writeFileSync(partialCandidatePath, "", "utf8");

const scoredRaw = readFileSync(partialRawPath, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
const scoredCandidate = readFileSync(partialCandidatePath, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));

const scoredRawIds = new Set(scoredRaw.map((r: any) => r.input_id));
const scoredCandidateIds = new Set(scoredCandidate.map((r: any) => r.source_candidate_id));

// --- Main loop (concurrent) ---
const queryIds = [...byInputId.keys()].sort();
let rawErrors = 0;
let candidateErrors = 0;
const concurrency = Math.max(1, parseInt(values.concurrency!, 10) || 4);

// Simple semaphore for concurrency control
function semaphore(limit: number) {
  let running = 0;
  const queue: (() => void)[] = [];
  const next = () => { if (queue.length > 0) { running++; queue.shift()!(); } };
  return {
  async run<T>(fn: () => Promise<T>): Promise<T> {
    if (running >= limit) await new Promise<void>(r => queue.push(r));
    running++;
    try { return await fn(); } finally { running--; next(); }
  },
  };
}
const pool = semaphore(concurrency);

// Mutex for file writes (appendFileSync is safe but we want consistent progress)
const writeLock = semaphore(1);

type BatchMeta =
  | { kind: "raw" }
  | { kind: "candidate"; scid: string; cid: string };

try {
  let completed = 0;
  const total = queryIds.length;

  await Promise.all(queryIds.map(inputId => pool.run(async () => {
    const candidates = byInputId.get(inputId)!;
    const first = candidates[0]!;
    const source = sources.get(first.source_id);
    if (!source) throw new Error(`Unknown source_id: ${first.source_id}`);

    // Collect all retrieval options for this query (raw + all candidates)
    const batchOptions: any[] = [];
    const batchMeta: BatchMeta[] = [];

    if (!scoredRawIds.has(inputId)) {
      batchOptions.push({
        originalQuery: first.query,
        expansions: [],
        collection: source.profile.collection_name,
        resultLimit: source.profile.result_limit,
        perListLimit: source.profile.per_list_limit,
        candidateLimit: source.profile.candidate_limit,
        rerank: false,
      });
      batchMeta.push({ kind: "raw" });
    }

    for (const candidate of candidates) {
      if (scoredCandidateIds.has(candidate.source_candidate_id)) continue;
      const expansions = candidate.canonical_output.map(
        ([type, text]: [string, string]) => ({ type, query: text })
      );
      batchOptions.push({
        originalQuery: candidate.query,
        expansions,
        collection: source.profile.collection_name,
        resultLimit: source.profile.result_limit,
        perListLimit: source.profile.per_list_limit,
        candidateLimit: source.profile.candidate_limit,
        rerank: false,
      });
      batchMeta.push({
        kind: "candidate",
        scid: candidate.source_candidate_id,
        cid: candidate.canonical_candidate_id,
      });
    }

    // Batch retrieve all at once
    if (batchOptions.length > 0) {
      try {
        const batchResults = await retrieveForBenchmarkBatch(store, batchOptions);

        // Process results
        for (let i = 0; i < batchMeta.length; i++) {
          const meta = batchMeta[i]!;
          const result = batchResults[i]!;

          // Convert to doc_ids
          const prefix = `qmd://${source.profile.collection_name}/`;
          const docIds: string[] = [];
          const seen = new Set<string>();
          for (const r of result) {
            if (!r.file.startsWith(prefix)) continue;
            const docId = source.docIdByPath.get(r.file.slice(prefix.length));
            if (!docId || seen.has(docId)) continue;
            seen.add(docId);
            docIds.push(docId);
          }
          const relevant = source.relevant.get(first.qid);
          if (!relevant?.size) throw new Error(`${first.qid}: no relevant documents`);
          const metrics = scoreCanonicalRanking(docIds, relevant, [1, 3, 5, 10, 20, 30]);

          if (meta.kind === "raw") {
            const rawRecord = {
              input_id: inputId,
              qid: first.qid,
              query: first.query,
              source_id: first.source_id,
              status: "scored",
              top_30_doc_ids: docIds,
              metrics,
            };
            await writeLock.run(async () => {
              appendFileSync(partialRawPath, `${JSON.stringify(rawRecord)}\n`, "utf8");
              scoredRawIds.add(inputId);
            });
          } else {
            const candRecord = {
              source_candidate_id: meta.scid,
              canonical_candidate_id: meta.cid,
              input_id: inputId,
              qid: first.qid,
              source_id: first.source_id,
              status: "scored",
              top_30_doc_ids: docIds,
              metrics,
            };
            await writeLock.run(async () => {
              appendFileSync(partialCandidatePath, `${JSON.stringify(candRecord)}\n`, "utf8");
              scoredCandidateIds.add(meta.scid);
            });
          }
        }
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        process.stderr.write(`  BATCH ERROR ${inputId}: ${msg}\n`);
        // Write error records for all items in the batch
        for (const meta of batchMeta) {
          if (meta.kind === "raw") {
            rawErrors++;
            await writeLock.run(async () => {
              appendFileSync(partialRawPath, `${JSON.stringify({ input_id: inputId, qid: first.qid, query: first.query, source_id: first.source_id, status: "error", error: msg })}\n`, "utf8");
              scoredRawIds.add(inputId);
            });
          } else {
            candidateErrors++;
            await writeLock.run(async () => {
              appendFileSync(partialCandidatePath, `${JSON.stringify({ source_candidate_id: meta.scid, canonical_candidate_id: meta.cid, input_id: inputId, qid: first.qid, source_id: first.source_id, status: "error", error: msg })}\n`, "utf8");
              scoredCandidateIds.add(meta.scid);
            });
          }
        }
      }
    }

    completed++;
    if (completed % 10 === 0 || completed === total) {
      process.stderr.write(
        `Query ${completed}/${total} | Raw ${scoredRawIds.size} | Candidates ${scoredCandidateIds.size}/${canonicalRecords.length}\n`
      );
    }
  })));
} finally {
  await llm.dispose();
  store.close();
}

// --- Finalize ---
renameSync(partialRawPath, rawPath);
renameSync(partialCandidatePath, candidatePath);

const rawRecords = readJsonl(rawPath);
const candidateRecords = readJsonl(candidatePath);

const rawScored = rawRecords.filter((r: any) => r.status === "scored");
const candidateScored = candidateRecords.filter((r: any) => r.status === "scored");

// --- Manifest ---
const manifest = {
  schema_version: "qmd-v2a-rescoring-v1",
  task: "2-retrieval-scoring",
  spec_section: "7",
  canonical_candidates_path: canonicalPath,
  canonical_candidates_sha256: sha256(readFileSync(canonicalPath)),
  retrieval_environment: {
    qmd_db_path: dbPath,
    qmd_db_sha256: sha256(readFileSync(dbPath)),
    rerank: false,
    auto_expand: false,
    result_limit: 30,
    per_list_limit: 30,
    candidate_limit: 40,
    index_manifests: indexManifests,
  },
  counts: {
    total_queries: byInputId.size,
    raw_scored: rawScored.length,
    raw_errors: rawErrors,
    total_candidates: canonicalRecords.length,
    candidate_scored: candidateScored.length,
    candidate_errors: candidateErrors,
  },
  outputs: {
    raw_retrieval: {
      path: rawPath,
      records: rawRecords.length,
      sha256: sha256(readFileSync(rawPath)),
    },
    candidate_retrieval: {
      path: candidatePath,
      records: candidateRecords.length,
      sha256: sha256(readFileSync(candidatePath)),
    },
  },
};
writeJson(manifestPath, manifest);

console.error(`\nDone.`);
console.error(`  Raw: ${rawScored.length} scored, ${rawErrors} errors`);
console.error(`  Candidates: ${candidateScored.length} scored, ${candidateErrors} errors`);
console.error(`  Manifest: ${manifestPath}`);
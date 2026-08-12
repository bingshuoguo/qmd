#!/usr/bin/env node

import {
  existsSync,
  readFileSync,
  renameSync,
  writeFileSync,
  appendFileSync,
} from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { homedir } from "node:os";
import {
  inspectBenchmarkIndex,
} from "../../../src/bench/bench.js";
import {
  loadBenchmarkV2,
  loadRetrievalProfile,
  parseDocumentsJsonl,
  parseQrelsTsv,
} from "../../../src/bench/qrels.js";
import { scoreCanonicalRanking } from "../../../src/bench/score.js";
import {
  createStore,
  retrieveForBenchmark,
} from "../../../src/store.js";
import { applyLlamaEnvMitigation, writeJson } from "../lib/cli.js";
import {
  assertQidPrefix,
  candidateIsAdmitted,
  hasRetrievalHeadroom,
  parseDistillRecords,
  parseSciFactDistillSplit,
  selectDistillWinner,
  sha256,
  type SciFactDistillRecord,
} from "../lib/distill.js";

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run distill:score -- --run-dir <dir> \\",
    "    [--source archive/scifact] \\",
    "    [--benchmark finetune/benchmarks/qmd-expansion-scifact-v1] [--db <index.sqlite>] \\",
    "    [--min-winners N]",
  ].join("\n");
  (exitCode === 0 ? console.log : console.error)(message);
  process.exit(exitCode);
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function renderProgress(options: {
  completed: number;
  total: number;
  resumed: number;
  startedAt: number;
  qid: string | null;
  status: string | null;
  winners: number;
}): void {
  if (!process.stderr.isTTY) return;
  const width = 24;
  const ratio = options.total === 0 ? 1 : Math.min(1, options.completed / options.total);
  const filled = Math.round(ratio * width);
  const elapsedMs = performance.now() - options.startedAt;
  const scoredThisRun = options.completed - options.resumed;
  const remaining = options.total - options.completed;
  const eta = scoredThisRun === 0
    ? "--"
    : formatDuration((elapsedMs / scoredThisRun) * remaining);
  const current = options.qid === null ? "" : ` qid=${options.qid}`;
  const status = options.status === null ? "" : ` status=${options.status}`;
  process.stderr.write(
    `\x1b[2K\rScore [${"=".repeat(filled)}${"-".repeat(width - filled)}]`
    + ` ${Math.round(ratio * 100).toString().padStart(3)}%`
    + ` ${options.completed}/${options.total}${current}${status}`
    + ` winners=${options.winners} elapsed=${formatDuration(elapsedMs)} eta=${eta}`
    + (options.completed === options.total ? "\n" : ""),
  );
}

function resultDocIds(
  files: readonly string[],
  collection: string,
  documentIdByPath: ReadonlyMap<string, string>,
): string[] {
  const prefix = `qmd://${collection}/`;
  const seen = new Set<string>();
  const docIds: string[] = [];
  for (const file of files) {
    if (!file.startsWith(prefix)) {
      throw new Error(`Result filepath is outside collection "${collection}": ${file}`);
    }
    const path = file.slice(prefix.length);
    const docId = documentIdByPath.get(path);
    if (!docId) throw new Error(`Result filepath is not in documents.jsonl: ${path}`);
    if (!seen.has(docId)) {
      seen.add(docId);
      docIds.push(docId);
    }
  }
  return docIds;
}

const { values } = parseArgs({
  options: {
    "run-dir": { type: "string" },
    source: { type: "string", default: "archive/scifact" },
    benchmark: {
      type: "string",
      default: "finetune/benchmarks/qmd-expansion-scifact-v1",
    },
    db: { type: "string" },
    "min-winners": { type: "string", default: "0" },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});
if (values.help) usage(0);
if (!values["run-dir"]) usage(1);
const minWinners = Number.parseInt(values["min-winners"], 10);
if (!Number.isSafeInteger(minWinners) || minWinners < 0) {
  throw new Error("--min-winners must be a non-negative integer");
}
const runDir = resolve(values["run-dir"]);
const sourceRoot = resolve(values.source);
const benchmarkRoot = resolve(values.benchmark);
const splitPath = resolve(runDir, "..", "..", "split.json");
const candidatesPath = join(runDir, "candidates.jsonl");
const partialPath = `${candidatesPath}.scored.partial`;
const manifestPath = join(runDir, "manifest.json");
const trainQrelsPath = join(sourceRoot, "qrels", "train.tsv");
const splitBytes = readFileSync(splitPath);
const candidatesBytes = readFileSync(candidatesPath);
const trainQrelsBytes = readFileSync(trainQrelsPath);
const split = parseSciFactDistillSplit(splitBytes.toString("utf8"));
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
if (manifest.split_sha256 !== sha256(splitBytes)) throw new Error("Run manifest split hash mismatch");
if (split.source.train_qrels_sha256 !== sha256(trainQrelsBytes)) {
  throw new Error("qrels/train.tsv does not match split.json");
}
const currentCandidatesSha = sha256(candidatesBytes);
if (manifest.scored_candidates_sha256 === currentCandidatesSha) {
  throw new Error("Candidate artifact has already been scored");
}
if (manifest.validated_candidates_sha256 !== currentCandidatesSha) {
  throw new Error("Candidates must pass Contract validation before scoring");
}

const benchmark = loadBenchmarkV2(benchmarkRoot);
if (benchmark.manifest.benchmark_id !== split.test_benchmark.benchmark_id) {
  throw new Error("Test benchmark id does not match split.json");
}
const benchmarkManifestBytes = readFileSync(join(benchmarkRoot, "benchmark.yaml"));
if (sha256(benchmarkManifestBytes) !== split.test_benchmark.benchmark_manifest_sha256) {
  throw new Error("Test benchmark manifest does not match split.json");
}
const profile = loadRetrievalProfile(benchmarkRoot, benchmark.manifest.cutoffs);
const frozenIndex = JSON.parse(
  readFileSync(join(benchmarkRoot, "index-manifest.json"), "utf8"),
);
const documentIdByPath = new Map(
  parseDocumentsJsonl(readFileSync(join(benchmarkRoot, "documents.jsonl"), "utf8"))
    .map(document => [document.path, document.doc_id]),
);
const relevantByQid = new Map<string, Set<string>>();
for (const qrel of parseQrelsTsv(trainQrelsBytes.toString("utf8"))) {
  if (qrel.relevance !== 1) continue;
  const relevant = relevantByQid.get(qrel.qid) ?? new Set<string>();
  relevant.add(qrel.doc_id);
  relevantByQid.set(qrel.qid, relevant);
}

const records = parseDistillRecords(candidatesBytes.toString("utf8"));
if (records.length === 0) throw new Error("candidates.jsonl must contain at least one query");
const expectedQids = [...split.train_qids, ...split.val_qids].map(qid => ({ qid }));
assertQidPrefix("Candidate artifact", records, expectedQids);
if (manifest.generated_queries !== records.length) {
  throw new Error("Candidate artifact length does not match the run manifest");
}
if (minWinners > records.length) {
  throw new Error(`--min-winners=${minWinners} exceeds ${records.length} scored queries`);
}
for (const record of records) {
  if (record.selection_status !== "pending" || record.raw_metrics !== null) {
    throw new Error(`qid ${record.qid}: candidate record was already scored`);
  }
  if (record.candidates.some(candidate => candidate.contract === null)) {
    throw new Error(`qid ${record.qid}: candidate is missing Contract validation`);
  }
}

if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");
const partialText = readFileSync(partialPath, "utf8");
const scored = partialText.trim() ? parseDistillRecords(partialText) : [];
assertQidPrefix("Scored partial artifact", scored, records);
const progressStartedAt = performance.now();
const resumedQueries = scored.length;
let progressLineOpen = false;
const reportProgress = (qid: string | null, status: string | null): void => {
  try {
    renderProgress({
      completed: scored.length,
      total: records.length,
      resumed: resumedQueries,
      startedAt: progressStartedAt,
      qid,
      status,
      winners: scored.filter(record => record.selection_status === "winner").length,
    });
    progressLineOpen = !!process.stderr.isTTY && scored.length < records.length;
  } catch {
    // Progress reporting must never fail or corrupt a scoring run.
  }
};
reportProgress(null, null);

const dbPath = resolve(
  values.db
    ?? process.env.INDEX_PATH
    ?? join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "qmd", "index.sqlite"),
);
if (!existsSync(dbPath)) throw new Error(`Benchmark index does not exist: ${dbPath}`);
applyLlamaEnvMitigation();
const { LlamaCpp } = await import("../../../src/llm.js");
const store = createStore(dbPath);
const llm = new LlamaCpp({
  embedModel: profile.embedding_model,
  rerankModel: profile.reranker_model ?? undefined,
  inactivityTimeoutMs: 5 * 60 * 1000,
  disposeModelsOnInactivity: true,
});
store.llm = llm;
try {
  const actualIndex = inspectBenchmarkIndex(store, benchmarkRoot, benchmark, profile);
  if (JSON.stringify(actualIndex) !== JSON.stringify(frozenIndex)) {
    throw new Error("Current collection/index does not match frozen index-manifest.json");
  }
  if (actualIndex.pending_embedding_count !== 0) {
    throw new Error(`Index has ${actualIndex.pending_embedding_count} documents pending embeddings`);
  }

  const score = async (
    record: SciFactDistillRecord,
    expansions: { type: "lex" | "vec" | "hyde"; query: string }[],
  ) => {
    const results = await retrieveForBenchmark(store, {
      originalQuery: record.query,
      expansions,
      collection: profile.collection_name,
      resultLimit: profile.result_limit,
      perListLimit: profile.per_list_limit,
      candidateLimit: profile.candidate_limit,
      rerank: profile.rerank,
    });
    const docIds = resultDocIds(
      results.map(result => result.file),
      profile.collection_name,
      documentIdByPath,
    );
    const relevant = relevantByQid.get(record.qid);
    if (!relevant?.size) throw new Error(`qid ${record.qid}: no relevant documents`);
    return scoreCanonicalRanking(docIds, relevant, benchmark.manifest.cutoffs);
  };

  for (const record of records.slice(scored.length)) {
    reportProgress(record.qid, "scoring");
    record.raw_metrics = await score(record, []);
    for (const candidate of record.candidates) {
      if (!candidateIsAdmitted(candidate)) continue;
      candidate.metrics = await score(
        record,
        candidate.contract!.canonical_output.map(([type, query]) => ({ type, query })),
      );
    }
    Object.assign(record, selectDistillWinner(record.candidates, record.raw_metrics));
    scored.push(record);
    appendFileSync(partialPath, `${JSON.stringify(record)}\n`, "utf8");
    reportProgress(record.qid, record.selection_status);
    if (!process.stderr.isTTY) {
      console.error(
        `Scored ${scored.length}/${records.length} qid=${record.qid} status=${record.selection_status}`,
      );
    }
  }
} finally {
  if (progressLineOpen) process.stderr.write("\n");
  await llm.dispose();
  store.close();
}

if (scored.length !== records.length) throw new Error("Candidate scoring did not complete");
renameSync(partialPath, candidatesPath);
manifest.retrieval = {
  benchmark_id: benchmark.manifest.benchmark_id,
  benchmark_manifest_sha256: sha256(benchmarkManifestBytes),
  profile_id: profile.profile_id,
  index_fingerprint: frozenIndex.index_fingerprint,
  winner_rule: "lexicographic(recall_at_30,mrr_at_10,ndcg_at_10); tie=min(candidate_index)",
};
manifest.scored_candidates_sha256 = sha256(readFileSync(candidatesPath));
const selectionCounts = Object.fromEntries(
  ["winner", "no_winner", "no_valid_candidate"].map(status => [
    status,
    scored.filter(record => record.selection_status === status).length,
  ]),
);
manifest.selection_counts = selectionCounts;
const winnerCount = selectionCounts.winner ?? 0;
const headroomQueries = scored.filter(hasRetrievalHeadroom).length;
const pilotGate = minWinners === 0
  ? null
  : {
      minimum_winners: minWinners,
      actual_winners: winnerCount,
      winner_rate: winnerCount / scored.length,
      headroom_queries: headroomQueries,
      passed: winnerCount >= minWinners,
    };
manifest.pilot_gate = pilotGate;
writeJson(manifestPath, manifest);
console.log(JSON.stringify({ run_dir: runDir, selection_counts: selectionCounts, pilot_gate: pilotGate }, null, 2));
if (pilotGate && minWinners > headroomQueries) {
  // Raised only after the manifest is persisted so the scoring work survives.
  throw new Error(
    `--min-winners=${minWinners} exceeds the ${headroomQueries} queries with retrieval headroom. `
    + "A winner must beat raw retrieval, and raw is already perfect on every other query, so no "
    + "teacher can satisfy this threshold. Lower it or score a wider query set.",
  );
}
if (pilotGate && !pilotGate.passed) process.exitCode = 2;

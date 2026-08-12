#!/usr/bin/env node

import {
  appendFileSync,
  existsSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { inspectBenchmarkIndex } from "../../../src/bench/bench.js";
import {
  loadBenchmarkV2,
  loadRetrievalProfile,
  parseDocumentsJsonl,
  parseQrelsTsv,
} from "../../../src/bench/qrels.js";
import { scoreCanonicalRanking } from "../../../src/bench/score.js";
import type { CanonicalQueryMetrics } from "../../../src/bench/types.js";
import { createStore, retrieveForBenchmark } from "../../../src/store.js";
import { applyLlamaEnvMitigation, writeJson } from "../lib/cli.js";
import {
  assertQidPrefix,
  candidateIsAdmitted,
  compareCanonicalMetrics,
  parseDistillRecords,
  parseSciFactDistillSplit,
  sha256,
  type SciFactDistillRecord,
} from "../lib/distill.js";

const ABLATION_MODES = ["lex_only", "vec_only", "hyde_only", "lex_vec"] as const;
const SUMMARY_MODES = [...ABLATION_MODES, "all"] as const;
type AblationMode = typeof ABLATION_MODES[number];
type SummaryMode = typeof SUMMARY_MODES[number];

type CandidateAblation = {
  candidate_index: number;
  metrics: Record<SummaryMode, CanonicalQueryMetrics | null>;
};

type AblationRecord = {
  qid: string;
  candidates: CandidateAblation[];
};

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run distill:ablate -- --run-dir <dir> \\",
    "    [--source archive/scifact] \\",
    "    [--benchmark finetune/benchmarks/qmd-expansion-scifact-v1] \\",
    "    [--db <index.sqlite>] [--force]",
    "",
    "Reuses scored raw/all metrics and runs lex-only, vec-only, HyDE-only, and lex+vec.",
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
  candidateIndex: number | null;
  mode: AblationMode | null;
}): void {
  if (!process.stderr.isTTY) return;
  const width = 24;
  const ratio = options.total === 0 ? 1 : Math.min(1, options.completed / options.total);
  const filled = Math.round(ratio * width);
  const elapsedMs = performance.now() - options.startedAt;
  const completedThisRun = options.completed - options.resumed;
  const remaining = options.total - options.completed;
  const eta = completedThisRun === 0
    ? "--"
    : formatDuration((elapsedMs / completedThisRun) * remaining);
  const current = options.qid === null
    ? ""
    : ` qid=${options.qid} candidate=${(options.candidateIndex ?? 0) + 1} mode=${options.mode}`;
  process.stderr.write(
    `\x1b[2K\rAblate [${"=".repeat(filled)}${"-".repeat(width - filled)}]`
    + ` ${Math.round(ratio * 100).toString().padStart(3)}%`
    + ` ${options.completed}/${options.total}${current}`
    + ` elapsed=${formatDuration(elapsedMs)} eta=${eta}`
    + (options.completed === options.total ? "\n" : ""),
  );
}

function parseAblationRecords(text: string): AblationRecord[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  return trimmed.split("\n").map((line, index) => {
    const value = JSON.parse(line) as AblationRecord;
    if (!value || typeof value.qid !== "string" || !Array.isArray(value.candidates)) {
      throw new Error(`Ablation partial line ${index + 1} is invalid`);
    }
    return value;
  });
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
    const docId = documentIdByPath.get(file.slice(prefix.length));
    if (!docId) throw new Error(`Result filepath is not in documents.jsonl: ${file}`);
    if (seen.has(docId)) continue;
    seen.add(docId);
    docIds.push(docId);
  }
  return docIds;
}

function expansionsFor(
  candidate: SciFactDistillRecord["candidates"][number],
  mode: AblationMode,
): { type: "lex" | "vec" | "hyde"; query: string }[] {
  const allowed = mode === "lex_vec"
    ? new Set(["lex", "vec"])
    : new Set([mode.replace("_only", "")]);
  return candidate.contract!.canonical_output
    .filter(([type]) => allowed.has(type))
    .map(([type, query]) => ({ type, query }));
}

function meanMetrics(metrics: readonly CanonicalQueryMetrics[]): CanonicalQueryMetrics {
  if (metrics.length === 0) throw new Error("Cannot average an empty metrics list");
  const keys = Object.keys(metrics[0]!) as (keyof CanonicalQueryMetrics)[];
  return Object.fromEntries(keys.map(key => [
    key,
    metrics.reduce((sum, item) => sum + item[key]!, 0) / metrics.length,
  ])) as unknown as CanonicalQueryMetrics;
}

function summarizeMode(
  sourceRecords: readonly SciFactDistillRecord[],
  ablationRecords: readonly AblationRecord[],
  mode: SummaryMode,
) {
  const selectionCounts = { winner: 0, no_winner: 0, no_valid_candidate: 0 };
  const candidateComparisons = { better: 0, tie: 0, worse: 0 };
  const selectedOrRaw: CanonicalQueryMetrics[] = [];

  for (let index = 0; index < sourceRecords.length; index++) {
    const source = sourceRecords[index]!;
    const raw = source.raw_metrics!;
    const metrics = ablationRecords[index]!.candidates
      .map(candidate => ({
        candidate_index: candidate.candidate_index,
        metrics: candidate.metrics[mode],
      }))
      .filter((item): item is { candidate_index: number; metrics: CanonicalQueryMetrics } => (
        item.metrics !== null
      ));

    for (const item of metrics) {
      const comparison = compareCanonicalMetrics(item.metrics, raw);
      candidateComparisons[comparison > 0 ? "better" : comparison < 0 ? "worse" : "tie"]++;
    }
    if (metrics.length === 0) {
      selectionCounts.no_valid_candidate++;
      selectedOrRaw.push(raw);
      continue;
    }
    const best = metrics.reduce((current, candidate) => {
      const comparison = compareCanonicalMetrics(candidate.metrics, current.metrics);
      if (comparison > 0) return candidate;
      if (comparison === 0 && candidate.candidate_index < current.candidate_index) return candidate;
      return current;
    });
    if (compareCanonicalMetrics(best.metrics, raw) > 0) {
      selectionCounts.winner++;
      selectedOrRaw.push(best.metrics);
    } else {
      selectionCounts.no_winner++;
      selectedOrRaw.push(raw);
    }
  }

  return {
    selection_counts: selectionCounts,
    candidate_comparisons: candidateComparisons,
    selected_or_raw_macro: meanMetrics(selectedOrRaw),
  };
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
    force: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});
if (values.help) usage(0);
if (!values["run-dir"]) usage(1);

const runDir = resolve(values["run-dir"]);
const sourceRoot = resolve(values.source);
const benchmarkRoot = resolve(values.benchmark);
const candidatesPath = join(runDir, "candidates.jsonl");
const manifestPath = join(runDir, "manifest.json");
const outputPath = join(runDir, "ablation.jsonl");
const partialPath = `${outputPath}.partial`;
const summaryPath = join(runDir, "ablation-summary.json");
const splitPath = resolve(runDir, "..", "..", "split.json");
const trainQrelsPath = join(sourceRoot, "qrels", "train.tsv");

if (values.force) {
  rmSync(outputPath, { force: true });
  rmSync(partialPath, { force: true });
  rmSync(summaryPath, { force: true });
} else if (existsSync(outputPath) || existsSync(summaryPath)) {
  throw new Error(`Ablation artifact already exists under ${runDir}; pass --force to replace it`);
}

const candidatesBytes = readFileSync(candidatesPath);
const splitBytes = readFileSync(splitPath);
const trainQrelsBytes = readFileSync(trainQrelsPath);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
if (manifest.scored_candidates_sha256 !== sha256(candidatesBytes)) {
  throw new Error("Candidates must be scored before ablation");
}
const split = parseSciFactDistillSplit(splitBytes.toString("utf8"));
if (manifest.split_sha256 !== sha256(splitBytes)) throw new Error("Run manifest split hash mismatch");
if (split.source.train_qrels_sha256 !== sha256(trainQrelsBytes)) {
  throw new Error("qrels/train.tsv does not match split.json");
}

const records = parseDistillRecords(candidatesBytes.toString("utf8"));
const expectedQids = [...split.train_qids, ...split.val_qids].map(qid => ({ qid }));
assertQidPrefix("Candidate artifact", records, expectedQids);
for (const record of records) {
  if (record.raw_metrics === null) throw new Error(`qid ${record.qid}: raw metrics are missing`);
  for (const candidate of record.candidates) {
    if (candidate.contract === null) throw new Error(`qid ${record.qid}: Contract result is missing`);
    if (candidateIsAdmitted(candidate) && candidate.metrics === null) {
      throw new Error(`qid ${record.qid}: scored metrics are missing`);
    }
  }
}

if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");
const completed = parseAblationRecords(readFileSync(partialPath, "utf8"));
assertQidPrefix("Ablation partial artifact", completed, records);
const completedByQid = new Map(completed.map(record => [record.qid, record]));
for (const record of completed) {
  const source = records.find(item => item.qid === record.qid)!;
  if (record.candidates.length !== source.candidates.length) {
    throw new Error(`qid ${record.qid}: ablation candidate count mismatch`);
  }
  for (let index = 0; index < source.candidates.length; index++) {
    const expected = source.candidates[index]!;
    const actual = record.candidates[index]!;
    if (actual.candidate_index !== expected.candidate_index) {
      throw new Error(`qid ${record.qid}: ablation candidate index mismatch`);
    }
    if (
      candidateIsAdmitted(expected)
      && SUMMARY_MODES.some(mode => actual.metrics[mode] === null)
    ) {
      throw new Error(`qid ${record.qid}: valid candidate has incomplete ablation metrics`);
    }
  }
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

const dbPath = resolve(
  values.db
    ?? process.env.INDEX_PATH
    ?? join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "qmd", "index.sqlite"),
);
if (!existsSync(dbPath)) throw new Error(`Benchmark index does not exist: ${dbPath}`);

const totalRetrievals = records.reduce(
  (sum, record) => sum + record.candidates.filter(candidateIsAdmitted).length,
  0,
) * ABLATION_MODES.length;
const resumedRetrievals = completed.reduce(
  (sum, record) => sum + record.candidates.filter(
    candidate => candidate.metrics.all !== null,
  ).length,
  0,
) * ABLATION_MODES.length;
let completedRetrievals = resumedRetrievals;
const startedAt = performance.now();
let progressLineOpen = false;
const reportProgress = (
  qid: string | null,
  candidateIndex: number | null,
  mode: AblationMode | null,
): void => {
  try {
    renderProgress({
      completed: completedRetrievals,
      total: totalRetrievals,
      resumed: resumedRetrievals,
      startedAt,
      qid,
      candidateIndex,
      mode,
    });
    progressLineOpen = !!process.stderr.isTTY && completedRetrievals < totalRetrievals;
  } catch {
    // Progress reporting must never fail an ablation run.
  }
};
reportProgress(null, null, null);

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

  for (const record of records) {
    if (completedByQid.has(record.qid)) continue;
    const candidates: CandidateAblation[] = [];
    for (const candidate of record.candidates) {
      const metrics = Object.fromEntries(
        SUMMARY_MODES.map(mode => [mode, mode === "all" ? candidate.metrics : null]),
      ) as Record<SummaryMode, CanonicalQueryMetrics | null>;
      if (candidateIsAdmitted(candidate)) {
        for (const mode of ABLATION_MODES) {
          metrics[mode] = await score(record, expansionsFor(candidate, mode));
          completedRetrievals++;
          reportProgress(record.qid, candidate.candidate_index, mode);
        }
      }
      candidates.push({ candidate_index: candidate.candidate_index, metrics });
    }
    const ablationRecord = { qid: record.qid, candidates };
    completed.push(ablationRecord);
    appendFileSync(partialPath, `${JSON.stringify(ablationRecord)}\n`, "utf8");
    if (!process.stderr.isTTY) {
      console.error(`Ablated ${completed.length}/${records.length} qid=${record.qid}`);
    }
  }
} finally {
  if (progressLineOpen) process.stderr.write("\n");
  await llm.dispose();
  store.close();
}

if (completed.length !== records.length) throw new Error("Ablation run did not complete");
renameSync(partialPath, outputPath);

const rawMacro = meanMetrics(records.map(record => record.raw_metrics!));
const modes = Object.fromEntries(SUMMARY_MODES.map(mode => {
  const summary = summarizeMode(records, completed, mode);
  return [mode, {
    ...summary,
    delta_vs_raw: Object.fromEntries(
      Object.keys(rawMacro).map(key => [
        key,
        summary.selected_or_raw_macro[key as keyof CanonicalQueryMetrics]!
          - rawMacro[key as keyof CanonicalQueryMetrics]!,
      ]),
    ),
  }];
}));
const expectedAllCounts = manifest.selection_counts;
if (JSON.stringify((modes.all as { selection_counts: unknown }).selection_counts) !== JSON.stringify(expectedAllCounts)) {
  throw new Error("All-mode ablation does not reproduce the scored selection counts");
}

const summary = {
  version: "scifact-distill-ablation-v1",
  completed_at: new Date().toISOString(),
  run_dir: runDir,
  candidates_sha256: sha256(candidatesBytes),
  benchmark_manifest_sha256: sha256(benchmarkManifestBytes),
  profile_id: profile.profile_id,
  index_fingerprint: frozenIndex.index_fingerprint,
  modes_evaluated: ["raw", ...SUMMARY_MODES],
  retrievals_required: totalRetrievals,
  retrievals_resumed: resumedRetrievals,
  retrievals_executed_this_run: totalRetrievals - resumedRetrievals,
  raw_macro: rawMacro,
  modes,
};
writeJson(summaryPath, summary);
console.log(JSON.stringify({
  run_dir: runDir,
  ablation_path: outputPath,
  summary_path: summaryPath,
  retrievals_required: totalRetrievals,
  retrievals_resumed: resumedRetrievals,
  retrievals_executed_this_run: totalRetrievals - resumedRetrievals,
  modes,
}, null, 2));

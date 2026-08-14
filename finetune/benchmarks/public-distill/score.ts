#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync, appendFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { homedir } from "node:os";
import { parseArgs } from "node:util";
import { inspectBenchmarkIndex } from "../../../src/bench/bench.js";
import { loadBenchmarkV2, loadRetrievalProfile, parseQrelsTsv } from "../../../src/bench/qrels.js";
import { scoreCanonicalRanking, averageCanonicalMetrics } from "../../../src/bench/score.js";
import { createStore, retrieveForBenchmark } from "../../../src/store.js";
import { applyLlamaEnvMitigation } from "../lib/cli.js";
import { selectPublicCandidate } from "../lib/public-distill.js";

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

const { values } = parseArgs({
  options: {
    "run-dir": { type: "string", default: "finetune/data/public-distill-v0/experiments/public-smoke-v0" },
    db: { type: "string" },
    "index-scope": { type: "string", default: "reduced" },
  },
});
const runDir = resolve(values["run-dir"]);
const publicRoot = resolve(runDir, "..", "..");
if (values["index-scope"] !== "reduced" && values["index-scope"] !== "formal") {
  throw new Error("--index-scope must be reduced or formal");
}
const diagnosticReducedIndex = values["index-scope"] === "reduced";
const preparedRoot = join(publicRoot, diagnosticReducedIndex ? "prepared-smoke-index" : "prepared");
const validatedPath = join(runDir, "validated.jsonl");
const scoredPath = join(runDir, "scored.jsonl");
const selectedPath = join(runDir, "selected.jsonl");
const partialPath = `${scoredPath}.partial`;
const manifestPath = join(runDir, "run-manifest.json");
if (existsSync(scoredPath) || existsSync(selectedPath)) throw new Error("Scored artifacts already exist");
const validatedBytes = readFileSync(validatedPath);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
if (manifest.validated_sha256 !== sha256(validatedBytes)) throw new Error("validated.jsonl hash mismatch");
if (manifest.generation_errors !== 0) throw new Error("Scoring requires generation_errors=0");
const records = validatedBytes.toString("utf8").trim().split("\n").map(line => JSON.parse(line));
const expectedRecords = manifest.smoke_only === true ? 30 : 2500;
if (records.length !== expectedRecords) {
  throw new Error(`Expected ${expectedRecords} validated records, got ${records.length}`);
}
if (manifest.smoke_only !== true && diagnosticReducedIndex) {
  throw new Error("Formal runs require --index-scope formal");
}

const sourceIds = ["fiqa-train", "cqadup-programmers", "cqadup-unix"];
const sources = new Map(sourceIds.map(sourceId => {
  const root = join(preparedRoot, sourceId);
  const benchmark = loadBenchmarkV2(root);
  const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
  const qrels = parseQrelsTsv(readFileSync(join(root, "qrels.tsv"), "utf8"));
  const relevant = new Map<string, Set<string>>();
  for (const qrel of qrels) {
    if (qrel.relevance !== 1) continue;
    const docs = relevant.get(qrel.qid) ?? new Set<string>();
    docs.add(qrel.doc_id);
    relevant.set(qrel.qid, docs);
  }
  return [sourceId, {
    root,
    benchmark,
    profile,
    relevant,
    docIdByPath: new Map(benchmark.documents.map(document => [document.path, document.doc_id])),
  }];
}));
const dbPath = resolve(values.db ?? join(runDir, "index.sqlite"));
if (!existsSync(dbPath)) throw new Error(`Experiment index does not exist: ${dbPath}`);
applyLlamaEnvMitigation();
const { LlamaCpp } = await import("../../../src/llm.js");
const firstProfile = sources.get(sourceIds[0]!)!.profile;
const store = createStore(dbPath);
const llm = new LlamaCpp({
  embedModel: firstProfile.embedding_model,
  rerankModel: firstProfile.reranker_model ?? undefined,
  inactivityTimeoutMs: 5 * 60 * 1000,
  disposeModelsOnInactivity: true,
});
store.llm = llm;
const indexManifests: Record<string, unknown> = {};
try {
  for (const [sourceId, source] of sources) {
    const frozen = JSON.parse(readFileSync(join(source.root, "index-manifest.json"), "utf8"));
    const actual = inspectBenchmarkIndex(store, source.root, source.benchmark, source.profile);
    if (JSON.stringify(actual) !== JSON.stringify(frozen)) {
      throw new Error(`${sourceId}: current index does not match frozen index manifest`);
    }
    if (actual.pending_embedding_count !== 0) throw new Error(`${sourceId}: embeddings are incomplete`);
    indexManifests[sourceId] = actual;
  }
  if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");
  const scored = readFileSync(partialPath, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
  for (let index = 0; index < scored.length; index++) {
    if (scored[index].input_id !== records[index]?.input_id) throw new Error("Scored partial is not a validated prefix");
  }
  const retrieve = async (record: any, expansions: [string, string][]) => {
    const source = sources.get(record.source_id);
    if (!source) throw new Error(`Unknown source_id: ${record.source_id}`);
    const results = await retrieveForBenchmark(store, {
      originalQuery: record.query,
      expansions: expansions.map(([type, query]) => ({ type: type as "lex" | "vec" | "hyde", query })),
      collection: source.profile.collection_name,
      resultLimit: source.profile.result_limit,
      perListLimit: source.profile.per_list_limit,
      candidateLimit: source.profile.candidate_limit,
      rerank: source.profile.rerank,
    });
    const prefix = `qmd://${source.profile.collection_name}/`;
    const docIds: string[] = [];
    const seen = new Set<string>();
    for (const result of results) {
      if (!result.file.startsWith(prefix)) throw new Error(`Cross-collection result: ${result.file}`);
      const docId = source.docIdByPath.get(result.file.slice(prefix.length));
      if (!docId) throw new Error(`Unknown result path: ${result.file}`);
      if (!seen.has(docId)) { seen.add(docId); docIds.push(docId); }
    }
    const relevant = source.relevant.get(record.qid);
    if (!relevant?.size) throw new Error(`${record.input_id}: no relevant documents`);
    return { top_30_doc_ids: docIds, metrics: scoreCanonicalRanking(docIds, relevant, [1, 3, 5, 10, 20, 30]) };
  };
  for (const record of records.slice(scored.length)) {
    try {
      const raw = await retrieve(record, []);
      record.raw = { status: "scored", ...raw };
      const admitted = [];
      for (const candidate of record.candidates) {
        const valid = candidate.generation_status === "ok"
          && candidate.contract?.valid === true
          && candidate.repeat_check?.valid === true;
        if (!valid) {
          candidate.retrieval = { status: "not_scored", top_30_doc_ids: [], metrics: null };
          continue;
        }
        const result = await retrieve(record, candidate.contract.canonical_output);
        candidate.retrieval = { status: "scored", ...result };
        admitted.push({ candidate_index: candidate.candidate_index, metrics: result.metrics });
      }
      Object.assign(record, selectPublicCandidate({
        sourceId: record.source_id,
        qid: record.qid,
        raw: raw.metrics,
        candidates: admitted,
      }));
    } catch (error) {
      record.selection_status = "retrieval_error";
      record.retrieval_error = error instanceof Error ? error.message : String(error);
      throw error;
    }
    scored.push(record);
    appendFileSync(partialPath, `${JSON.stringify(record)}\n`, "utf8");
    process.stderr.write(`Scored ${scored.length}/${expectedRecords} ${record.input_id}: ${record.selection_status}\n`);
  }
  renameSync(partialPath, scoredPath);
  writeFileSync(selectedPath, readFileSync(scoredPath));
  const counts = Object.fromEntries(
    ["winner", "qualified_tie", "no_winner", "no_valid_candidate", "retrieval_error"]
      .map(status => [status, scored.filter(record => record.selection_status === status).length]),
  );
  const accepted = scored.filter(record => ["winner", "qualified_tie"].includes(record.selection_status));
  const rawMetrics = averageCanonicalMetrics(scored.map(record => record.raw.metrics), [1, 3, 5, 10, 20, 30]);
  const selectedMetrics = accepted.length > 0
    ? averageCanonicalMetrics(accepted.map(record => (
      record.candidates[record.selected_candidate_index].retrieval.metrics
    )), [1, 3, 5, 10, 20, 30])
    : null;
  manifest.index_manifests = indexManifests;
  manifest.scored_sha256 = sha256(readFileSync(scoredPath));
  manifest.selected_sha256 = sha256(readFileSync(selectedPath));
  manifest.selection_counts = counts;
  manifest.retrieval_profile = {
    result_limit: 30,
    per_list_limit: 30,
    candidate_limit: 40,
    rerank: true,
    auto_generate_expansions: false,
    selection_order: ["recall_at_10", "ndcg_at_10", "mrr_at_10"],
    diagnostic_reduced_index: diagnosticReducedIndex,
    final_sft_eligible: false,
  };
  writeJson(manifestPath, manifest);
  writeJson(join(runDir, "report.json"), {
    schema_version: "qmd-public-distill-report-v0",
    smoke_only: manifest.smoke_only === true,
    diagnostic_reduced_index: diagnosticReducedIndex,
    final_sft_eligible: false,
    input_queries: scored.length,
    candidate_count: scored.length * 4,
    generation_errors: manifest.generation_errors,
    contract_invalid: scored.flatMap(record => record.candidates).filter(candidate => candidate.contract?.valid !== true).length,
    repeat_only: scored.flatMap(record => record.candidates).filter(candidate => candidate.repeat_check?.repeat_only === true).length,
    retrieval_errors: counts.retrieval_error,
    selection_counts: counts,
    raw_aggregate: rawMetrics,
    accepted_aggregate: selectedMetrics,
    note: "Oracle selection metrics validate data production only; they are not model gains.",
  });
} finally {
  await llm.dispose();
  store.close();
}

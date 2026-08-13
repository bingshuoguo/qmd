import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { basename, join, resolve } from "node:path";
import YAML from "yaml";
import {
  QMD_EXPANSION_SCIFACT_BENCHMARK_ID,
  type BenchmarkDocument,
  type BenchmarkExpansion,
  type BenchmarkExpansionRecord,
  type BenchmarkManifestV2,
  type BenchmarkQrel,
  type BenchmarkQueryV2,
  type CanonicalMetric,
  type ExpansionStatus,
  type ExpansionType,
  type LoadedBenchmarkV2,
  type RetrievalProfileV2,
} from "./types.js";

const ALLOWED_METRICS = new Set<CanonicalMetric>([
  "recall_at_cutoffs",
  "mrr_at_10",
  "ndcg_at_10",
]);
const ALLOWED_EXPANSION_TYPES = new Set<ExpansionType>(["lex", "vec", "hyde"]);
const ALLOWED_EXPANSION_STATUSES = new Set<ExpansionStatus>([
  "ok",
  "format_error",
  "generation_error",
]);

function record(value: unknown, location: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${location}: expected an object`);
  }
  return value as Record<string, unknown>;
}

function nonEmptyString(value: unknown, location: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${location}: expected a non-empty string`);
  }
  return value;
}

function parseJsonLines(text: string, label: string): unknown[] {
  const values: unknown[] = [];
  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!;
    if (line.trim().length === 0) continue;
    try {
      values.push(JSON.parse(line));
    } catch (error) {
      throw new Error(
        `${label}:${index + 1}: invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  return values;
}

export function parseQueriesJsonl(text: string): BenchmarkQueryV2[] {
  const queries: BenchmarkQueryV2[] = [];
  const seen = new Set<string>();
  for (const [index, value] of parseJsonLines(text, "queries.jsonl").entries()) {
    const item = record(value, `queries.jsonl:${index + 1}`);
    const qid = nonEmptyString(item.qid, `queries.jsonl:${index + 1}.qid`);
    const query = nonEmptyString(item.query, `queries.jsonl:${index + 1}.query`);
    if (seen.has(qid)) throw new Error(`queries.jsonl: duplicate qid "${qid}"`);
    seen.add(qid);
    queries.push({ qid, query });
  }
  if (queries.length === 0) throw new Error("queries.jsonl: no queries");
  return queries;
}

export function parseDocumentsJsonl(text: string): BenchmarkDocument[] {
  const documents: BenchmarkDocument[] = [];
  const docIds = new Set<string>();
  const paths = new Set<string>();
  for (const [index, value] of parseJsonLines(text, "documents.jsonl").entries()) {
    const item = record(value, `documents.jsonl:${index + 1}`);
    const doc_id = nonEmptyString(item.doc_id, `documents.jsonl:${index + 1}.doc_id`);
    const path = nonEmptyString(item.path, `documents.jsonl:${index + 1}.path`);
    if (docIds.has(doc_id)) throw new Error(`documents.jsonl: duplicate doc_id "${doc_id}"`);
    if (paths.has(path)) throw new Error(`documents.jsonl: duplicate path "${path}"`);
    docIds.add(doc_id);
    paths.add(path);
    documents.push({ doc_id, path });
  }
  if (documents.length === 0) throw new Error("documents.jsonl: no documents");
  return documents;
}

export function parseQrelsTsv(text: string): BenchmarkQrel[] {
  const lines = text.split(/\r?\n/);
  if (lines[0] !== "query-id\tcorpus-id\tscore") {
    throw new Error("qrels.tsv: expected BEIR header query-id<TAB>corpus-id<TAB>score");
  }

  const qrels: BenchmarkQrel[] = [];
  const seen = new Map<string, 0 | 1>();
  for (let index = 1; index < lines.length; index++) {
    const line = lines[index]!;
    if (line.length === 0) continue;
    const columns = line.split("\t");
    if (columns.length !== 3) {
      throw new Error(`qrels.tsv:${index + 1}: expected exactly three tab-separated columns`);
    }
    const [qidValue, docIdValue, relevanceValue] = columns;
    const qid = nonEmptyString(qidValue, `qrels.tsv:${index + 1}.query-id`);
    const doc_id = nonEmptyString(docIdValue, `qrels.tsv:${index + 1}.corpus-id`);
    if (relevanceValue !== "0" && relevanceValue !== "1") {
      throw new Error(`qrels.tsv:${index + 1}: relevance must be 0 or 1`);
    }
    const relevance = Number(relevanceValue) as 0 | 1;
    const key = `${qid}\0${doc_id}`;
    const previous = seen.get(key);
    if (previous !== undefined) {
      const kind = previous === relevance ? "duplicate" : "conflicting";
      throw new Error(`qrels.tsv:${index + 1}: ${kind} qrel for qid "${qid}", doc_id "${doc_id}"`);
    }
    seen.set(key, relevance);
    qrels.push({ qid, doc_id, relevance });
  }
  if (qrels.length === 0) throw new Error("qrels.tsv: no qrels");
  return qrels;
}

export function validateBenchmarkData(
  queries: BenchmarkQueryV2[],
  qrels: BenchmarkQrel[],
  documents: BenchmarkDocument[],
): Omit<LoadedBenchmarkV2, "manifest" | "queries" | "qrels" | "documents"> {
  const queryById = new Map(queries.map(query => [query.qid, query]));
  const documentById = new Map(documents.map(document => [document.doc_id, document]));
  const relevanceByQuery = new Map<string, Map<string, 0 | 1>>();

  for (const qrel of qrels) {
    if (!queryById.has(qrel.qid)) {
      throw new Error(`qrels.tsv: unknown qid "${qrel.qid}"`);
    }
    if (!documentById.has(qrel.doc_id)) {
      throw new Error(`qrels.tsv: unknown doc_id "${qrel.doc_id}"`);
    }
    let relevance = relevanceByQuery.get(qrel.qid);
    if (!relevance) {
      relevance = new Map();
      relevanceByQuery.set(qrel.qid, relevance);
    }
    if (relevance.has(qrel.doc_id)) {
      throw new Error(`qrels.tsv: duplicate qrel for qid "${qrel.qid}", doc_id "${qrel.doc_id}"`);
    }
    relevance.set(qrel.doc_id, qrel.relevance);
  }

  for (const query of queries) {
    const relevance = relevanceByQuery.get(query.qid);
    if (!relevance || !Array.from(relevance.values()).some(value => value === 1)) {
      throw new Error(`query "${query.qid}" has no relevant documents`);
    }
  }

  return { queryById, documentById, relevanceByQuery };
}

export function validateBenchmarkManifest(
  value: unknown,
  expectedBenchmarkId = QMD_EXPANSION_SCIFACT_BENCHMARK_ID,
): BenchmarkManifestV2 {
  const manifest = record(value, "benchmark.yaml");
  if (manifest.benchmark_id !== expectedBenchmarkId) {
    throw new Error(
      `benchmark.yaml: benchmark_id must be "${expectedBenchmarkId}", got ${JSON.stringify(manifest.benchmark_id)}`,
    );
  }
  const source = record(manifest.source, "benchmark.yaml.source");
  nonEmptyString(source.url, "benchmark.yaml.source.url");
  if (typeof source.archive_md5 !== "string" || !/^[a-f0-9]{32}$/.test(source.archive_md5)) {
    throw new Error("benchmark.yaml: source.archive_md5 must be a lowercase MD5");
  }
  nonEmptyString(source.split, "benchmark.yaml.source.split");
  for (const key of [
    "source_qrels_sha256",
    "excluded_qids_sha256",
    "leakage_report_sha256",
    "converted_data_sha256",
  ] as const) {
    if (typeof manifest[key] !== "string" || !/^[a-f0-9]{64}$/.test(manifest[key])) {
      throw new Error(`benchmark.yaml: ${key} must be a lowercase SHA256`);
    }
  }
  const qrels = record(manifest.qrels, "benchmark.yaml.qrels");
  if (
    qrels.relevant_threshold !== 1
    || qrels.unjudged !== "nonrelevant"
    || qrels.graded !== false
  ) {
    throw new Error("benchmark.yaml: qrels must be binary with threshold 1 and unjudged nonrelevant");
  }
  if (!Array.isArray(manifest.cutoffs) || manifest.cutoffs.length === 0) {
    throw new Error("benchmark.yaml: cutoffs must be a non-empty array");
  }
  const seenCutoffs = new Set<number>();
  for (const cutoff of manifest.cutoffs) {
    if (!Number.isSafeInteger(cutoff) || cutoff <= 0) {
      throw new Error(`benchmark.yaml: invalid cutoff ${JSON.stringify(cutoff)}`);
    }
    if (seenCutoffs.has(cutoff)) {
      throw new Error(`benchmark.yaml: duplicate cutoff ${cutoff}`);
    }
    seenCutoffs.add(cutoff);
  }
  if (!Array.isArray(manifest.metrics) || manifest.metrics.length === 0) {
    throw new Error("benchmark.yaml: metrics must be a non-empty array");
  }
  const seenMetrics = new Set<string>();
  for (const metric of manifest.metrics) {
    if (typeof metric !== "string" || !ALLOWED_METRICS.has(metric as CanonicalMetric)) {
      throw new Error(`benchmark.yaml: unknown metric ${JSON.stringify(metric)}`);
    }
    if (seenMetrics.has(metric)) {
      throw new Error(`benchmark.yaml: duplicate metric ${JSON.stringify(metric)}`);
    }
    seenMetrics.add(metric);
  }
  return manifest as unknown as BenchmarkManifestV2;
}

export function loadBenchmarkV2(benchmarkDir: string): LoadedBenchmarkV2 {
  const root = resolve(benchmarkDir);
  const expectedBenchmarkId = basename(root);
  const manifest = validateBenchmarkManifest(
    YAML.parse(readFileSync(join(root, "benchmark.yaml"), "utf8")),
    expectedBenchmarkId,
  );
  const sourceQrelsBytes = readFileSync(join(root, "source-qrels.tsv"));
  const excludedBytes = readFileSync(join(root, "excluded-qids.json"));
  const leakageBytes = readFileSync(join(root, "leakage-report.json"));
  const digest = (bytes: Uint8Array): string =>
    createHash("sha256").update(bytes).digest("hex");
  if (digest(sourceQrelsBytes) !== manifest.source_qrels_sha256) {
    throw new Error("source-qrels.tsv SHA256 does not match benchmark.yaml");
  }
  if (digest(excludedBytes) !== manifest.excluded_qids_sha256) {
    throw new Error("excluded-qids.json SHA256 does not match benchmark.yaml");
  }
  if (digest(leakageBytes) !== manifest.leakage_report_sha256) {
    throw new Error("leakage-report.json SHA256 does not match benchmark.yaml");
  }
  const excluded = JSON.parse(excludedBytes.toString("utf8"));
  if (
    !Array.isArray(excluded)
    || excluded.some(qid => typeof qid !== "string")
    || new Set(excluded).size !== excluded.length
  ) {
    throw new Error("excluded-qids.json must be an array of unique string qids");
  }
  const excludedSet = new Set<string>(excluded);
  const sourceQrelsText = sourceQrelsBytes.toString("utf8");
  const expectedQrels = `${sourceQrelsText.split(/\r?\n/)
    .filter((line, index) =>
      index === 0 || (line.length > 0 && !excludedSet.has(line.split("\t")[0]!))
    )
    .join("\n")}\n`;
  const qrelsText = readFileSync(join(root, "qrels.tsv"), "utf8");
  if (qrelsText !== expectedQrels) {
    throw new Error("qrels.tsv must equal source-qrels.tsv with only excluded qids removed");
  }
  const queries = parseQueriesJsonl(readFileSync(join(root, "queries.jsonl"), "utf8"));
  const qrels = parseQrelsTsv(qrelsText);
  const documents = parseDocumentsJsonl(readFileSync(join(root, "documents.jsonl"), "utf8"));
  const indexes = validateBenchmarkData(queries, qrels, documents);
  return { manifest, queries, qrels, documents, ...indexes };
}

export function validateRetrievalProfile(
  value: unknown,
  cutoffs: readonly number[],
): RetrievalProfileV2 {
  const profile = record(value, "retrieval-profile.yaml");
  const integer = (key: "result_limit" | "per_list_limit" | "candidate_limit"): number => {
    const candidate = profile[key];
    if (!Number.isSafeInteger(candidate) || (candidate as number) <= 0) {
      throw new Error(`retrieval-profile.yaml: ${key} must be a positive integer`);
    }
    return candidate as number;
  };
  const resultLimit = integer("result_limit");
  const perListLimit = integer("per_list_limit");
  const candidateLimit = integer("candidate_limit");
  const maxCutoff = Math.max(...cutoffs);
  if (resultLimit < maxCutoff) {
    throw new Error(`retrieval-profile.yaml: result_limit must be at least ${maxCutoff}`);
  }
  if (perListLimit < resultLimit) {
    throw new Error("retrieval-profile.yaml: per_list_limit must be at least result_limit");
  }
  if (candidateLimit < resultLimit) {
    throw new Error("retrieval-profile.yaml: candidate_limit must be at least result_limit");
  }
  if (profile.auto_expand !== false) {
    throw new Error("retrieval-profile.yaml: auto_expand must be false");
  }
  if (profile.strong_signal_bypass !== false) {
    throw new Error("retrieval-profile.yaml: strong_signal_bypass must be false");
  }
  if (typeof profile.rerank !== "boolean") {
    throw new Error("retrieval-profile.yaml: rerank must be boolean");
  }
  nonEmptyString(profile.profile_id, "retrieval-profile.yaml.profile_id");
  nonEmptyString(profile.collection_name, "retrieval-profile.yaml.collection_name");
  nonEmptyString(profile.collection_root, "retrieval-profile.yaml.collection_root");
  nonEmptyString(profile.embedding_model, "retrieval-profile.yaml.embedding_model");
  if (profile.reranker_model !== null && typeof profile.reranker_model !== "string") {
    throw new Error("retrieval-profile.yaml: reranker_model must be a string or null");
  }
  if (profile.rerank && !profile.reranker_model) {
    throw new Error("retrieval-profile.yaml: reranker_model is required when rerank is true");
  }
  return {
    profile_id: profile.profile_id as string,
    collection_name: profile.collection_name as string,
    collection_root: profile.collection_root as string,
    embedding_model: profile.embedding_model as string,
    reranker_model: profile.reranker_model as string | null,
    result_limit: resultLimit,
    per_list_limit: perListLimit,
    candidate_limit: candidateLimit,
    rerank: profile.rerank,
    auto_expand: false,
    strong_signal_bypass: false,
  };
}

export function loadRetrievalProfile(
  benchmarkDir: string,
  cutoffs: readonly number[],
): RetrievalProfileV2 {
  return validateRetrievalProfile(
    YAML.parse(readFileSync(join(resolve(benchmarkDir), "retrieval-profile.yaml"), "utf8")),
    cutoffs,
  );
}

function parseExpansion(value: unknown, location: string): BenchmarkExpansion {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`${location}: expansion must be a [type, text] pair`);
  }
  const [type, text] = value;
  if (typeof type !== "string" || !ALLOWED_EXPANSION_TYPES.has(type as ExpansionType)) {
    throw new Error(`${location}: invalid expansion type ${JSON.stringify(type)}`);
  }
  if (typeof text !== "string" || text.trim().length === 0) {
    throw new Error(`${location}: expansion text must be non-empty`);
  }
  if (/[\r\n]/.test(text)) {
    throw new Error(`${location}: expansion text must be single-line`);
  }
  return [type as ExpansionType, text];
}

export function parseExpansionsJsonl(
  text: string,
  queries: BenchmarkQueryV2[],
): BenchmarkExpansionRecord[] {
  const queryById = new Map(queries.map(query => [query.qid, query]));
  const expansions: BenchmarkExpansionRecord[] = [];
  const seen = new Set<string>();

  for (const [index, value] of parseJsonLines(text, "expansions.jsonl").entries()) {
    const location = `expansions.jsonl:${index + 1}`;
    const item = record(value, location);
    const qid = nonEmptyString(item.qid, `${location}.qid`);
    if (seen.has(qid)) throw new Error(`expansions.jsonl: duplicate qid "${qid}"`);
    seen.add(qid);
    const benchmarkQuery = queryById.get(qid);
    if (!benchmarkQuery) throw new Error(`expansions.jsonl: extra qid "${qid}"`);
    if (item.query !== benchmarkQuery.query) {
      throw new Error(`expansions.jsonl: query mismatch for qid "${qid}"`);
    }
    if (
      typeof item.status !== "string"
      || !ALLOWED_EXPANSION_STATUSES.has(item.status as ExpansionStatus)
    ) {
      throw new Error(`${location}: invalid status ${JSON.stringify(item.status)}`);
    }
    if (!Array.isArray(item.output)) throw new Error(`${location}.output: expected an array`);
    const output = item.output.map((entry, outputIndex) =>
      parseExpansion(entry, `${location}.output[${outputIndex}]`)
    );
    if (item.status !== "ok" && output.length > 0) {
      throw new Error(`${location}: non-ok expansion must use empty output`);
    }
    if (typeof item.raw_output !== "string") {
      throw new Error(`${location}.raw_output: expected a string`);
    }
    if (typeof item.fallback_used !== "boolean") {
      throw new Error(`${location}.fallback_used: expected a boolean`);
    }
    if (item.error !== null && typeof item.error !== "string") {
      throw new Error(`${location}.error: expected a string or null`);
    }
    if (
      item.status !== "ok"
      && (typeof item.error !== "string" || item.error.length === 0)
    ) {
      throw new Error(`${location}: non-ok expansion must include an error reason`);
    }
    if (item.status !== "ok" && item.fallback_used !== false) {
      throw new Error(`${location}: non-ok expansion must set fallback_used to false`);
    }
    if (item.status === "ok" && item.error !== null) {
      throw new Error(`${location}: ok expansion must use error: null`);
    }
    expansions.push({
      qid,
      query: benchmarkQuery.query,
      status: item.status as ExpansionStatus,
      raw_output: item.raw_output,
      output,
      fallback_used: item.fallback_used,
      error: item.error,
    });
  }

  for (const query of queries) {
    if (!seen.has(query.qid)) {
      throw new Error(`expansions.jsonl: missing qid "${query.qid}"`);
    }
  }
  return expansions;
}

import { createHash } from "node:crypto";
import type { BenchmarkExpansion, CanonicalQueryMetrics } from "../../../src/bench/types.js";
import type {
  OpenAiGenerationDiagnostics,
  OpenAiGenerationErrorCode,
} from "./openai-teacher.js";

export const SCIFACT_DISTILL_SPLIT_VERSION = "scifact-distill-split-v1";
export const SCIFACT_DISTILL_ALGORITHM_VERSION = "sha256-qid-v1";
export const SCIFACT_DISTILL_NORMALIZATION_VERSION = "nfkc-lower-whitespace-v1";
export const SCIFACT_DISTILL_SEED = 42;
export const SCIFACT_DISTILL_VALIDATION_COUNT = 161;
export const SCIFACT_DISTILL_CANDIDATE_COUNT = 4;

export type SciFactSourceQuery = {
  qid: string;
  query: string;
};

export type SciFactQrel = {
  qid: string;
  doc_id: string;
  relevance: 0 | 1;
};

export type SciFactDistillExclusion = {
  qid: string;
  reason: "exact_test_query_overlap" | "duplicate_train_query";
  duplicate_of_qid: string;
};

export type SciFactDistillSplit = {
  version: typeof SCIFACT_DISTILL_SPLIT_VERSION;
  dataset: "scifact-v1";
  source: {
    archive_md5: string;
    queries_sha256: string;
    train_qrels_sha256: string;
    test_qrels_sha256: string;
  };
  test_benchmark: {
    benchmark_id: string;
    benchmark_manifest_sha256: string;
  };
  algorithm: {
    version: typeof SCIFACT_DISTILL_ALGORITHM_VERSION;
    normalization: typeof SCIFACT_DISTILL_NORMALIZATION_VERSION;
    seed: typeof SCIFACT_DISTILL_SEED;
    validation_count: typeof SCIFACT_DISTILL_VALIDATION_COUNT;
  };
  counts: {
    source_train_queries: number;
    source_test_queries: number;
    excluded_queries: number;
    train_queries: number;
    val_queries: number;
  };
  exclusions: SciFactDistillExclusion[];
  train_qids: string[];
  val_qids: string[];
};

export type DistillContractResult = {
  version: string;
  valid: boolean;
  errors: { code: string; path: string; message: string }[];
  warnings: { code: string; path: string; message: string }[];
  canonical_output: BenchmarkExpansion[];
};

export type DistillSemanticGateResult = {
  version: string;
  valid: boolean;
  errors: { code: string; path: string; message: string }[];
  /** Recorded for audit only; never affects `valid` or winner selection. */
  advisories?: { code: string; path: string; message: string }[];
};

export type DistillCandidate = {
  candidate_index: number;
  generation_status: "ok" | "format_error" | "generation_error";
  raw_output: string;
  parsed_output: BenchmarkExpansion[];
  generation_error: string | null;
  generation_error_type?: OpenAiGenerationErrorCode | null;
  generation_diagnostics?: OpenAiGenerationDiagnostics | null;
  contract: DistillContractResult | null;
  semantic_gate?: DistillSemanticGateResult | null;
  metrics: CanonicalQueryMetrics | null;
};

export type DistillSelectionStatus =
  | "pending"
  | "winner"
  | "no_winner"
  | "no_valid_candidate";

export type SciFactDistillRecord = {
  qid: string;
  split: "train" | "val";
  query: string;
  raw_metrics: CanonicalQueryMetrics | null;
  candidates: DistillCandidate[];
  selected_candidate_index: number | null;
  selection_status: DistillSelectionStatus;
};

function byteCompare(left: string, right: string): number {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

export function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

export function normalizeSciFactQuery(query: string): string {
  return query.normalize("NFKC").toLowerCase().trim().replace(/\s+/g, " ");
}

export function sciFactSplitKey(qid: string): string {
  return sha256(`scifact-v1\0seed=${SCIFACT_DISTILL_SEED}\0${qid}`);
}

export function parseSciFactSourceQueries(text: string): SciFactSourceQuery[] {
  const queries: SciFactSourceQuery[] = [];
  const seen = new Set<string>();
  for (const [index, rawLine] of text.split(/\r?\n/).entries()) {
    if (!rawLine.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(rawLine);
    } catch (error) {
      throw new Error(`queries.jsonl:${index + 1}: invalid JSON`, { cause: error });
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`queries.jsonl:${index + 1}: record must be an object`);
    }
    const record = value as Record<string, unknown>;
    if (typeof record._id !== "string" || !record._id) {
      throw new Error(`queries.jsonl:${index + 1}: _id must be a non-empty string`);
    }
    if (typeof record.text !== "string" || !record.text.trim()) {
      throw new Error(`queries.jsonl:${index + 1}: text must be a non-empty string`);
    }
    if (seen.has(record._id)) throw new Error(`queries.jsonl: duplicate _id "${record._id}"`);
    seen.add(record._id);
    queries.push({ qid: record._id, query: record.text });
  }
  return queries;
}

export function parseSciFactDistillSplit(text: string): SciFactDistillSplit {
  const value = JSON.parse(text) as Partial<SciFactDistillSplit>;
  if (value.version !== SCIFACT_DISTILL_SPLIT_VERSION) {
    throw new Error(`split.json: unsupported version ${JSON.stringify(value.version)}`);
  }
  if (
    value.algorithm?.version !== SCIFACT_DISTILL_ALGORITHM_VERSION
    || value.algorithm.normalization !== SCIFACT_DISTILL_NORMALIZATION_VERSION
    || value.algorithm.seed !== SCIFACT_DISTILL_SEED
    || value.algorithm.validation_count !== SCIFACT_DISTILL_VALIDATION_COUNT
  ) {
    throw new Error("split.json: algorithm contract mismatch");
  }
  if (!Array.isArray(value.train_qids) || !Array.isArray(value.val_qids)) {
    throw new Error("split.json: train_qids and val_qids must be arrays");
  }
  if (
    value.train_qids.length !== value.counts?.train_queries
    || value.val_qids.length !== value.counts?.val_queries
  ) {
    throw new Error("split.json: qid arrays do not match declared counts");
  }
  const all = [...value.train_qids, ...value.val_qids];
  if (all.some(qid => typeof qid !== "string" || !qid)) {
    throw new Error("split.json: every qid must be a non-empty string");
  }
  if (new Set(all).size !== all.length) throw new Error("split.json: qids overlap or repeat");
  return value as SciFactDistillSplit;
}

export function parseDistillRecords(text: string): SciFactDistillRecord[] {
  const records: SciFactDistillRecord[] = [];
  const seen = new Set<string>();
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    let record: SciFactDistillRecord;
    try {
      record = JSON.parse(line) as SciFactDistillRecord;
    } catch (error) {
      throw new Error(`candidates.jsonl:${index + 1}: invalid JSON`, { cause: error });
    }
    validateDistillRecordShape(record);
    if (seen.has(record.qid)) throw new Error(`candidates.jsonl: duplicate qid "${record.qid}"`);
    seen.add(record.qid);
    records.push(record);
  }
  return records;
}

function qidsFromQrels(qrels: readonly SciFactQrel[]): Set<string> {
  return new Set(qrels.map(qrel => qrel.qid));
}

export function buildSciFactDistillMembership(
  queries: readonly SciFactSourceQuery[],
  trainQrels: readonly SciFactQrel[],
  testQrels: readonly SciFactQrel[],
): Pick<SciFactDistillSplit, "counts" | "exclusions" | "train_qids" | "val_qids"> {
  const queryById = new Map(queries.map(query => [query.qid, query.query]));
  const trainQids = qidsFromQrels(trainQrels);
  const testQids = qidsFromQrels(testQrels);
  for (const qid of new Set([...trainQids, ...testQids])) {
    if (!queryById.has(qid)) throw new Error(`qrels reference missing query "${qid}"`);
  }
  const directOverlap = [...trainQids].filter(qid => testQids.has(qid));
  if (directOverlap.length > 0) {
    throw new Error(`train/test qid overlap: ${directOverlap.sort(byteCompare).join(", ")}`);
  }

  const testByNormalized = new Map<string, string[]>();
  for (const qid of testQids) {
    const key = normalizeSciFactQuery(queryById.get(qid)!);
    const group = testByNormalized.get(key) ?? [];
    group.push(qid);
    testByNormalized.set(key, group);
  }
  for (const group of testByNormalized.values()) group.sort(byteCompare);

  const exclusions: SciFactDistillExclusion[] = [];
  const afterTestLeakage: string[] = [];
  for (const qid of [...trainQids].sort(byteCompare)) {
    const duplicateTestQids = testByNormalized.get(normalizeSciFactQuery(queryById.get(qid)!));
    if (duplicateTestQids) {
      exclusions.push({
        qid,
        reason: "exact_test_query_overlap",
        duplicate_of_qid: duplicateTestQids[0]!,
      });
    } else {
      afterTestLeakage.push(qid);
    }
  }

  const trainByNormalized = new Map<string, string[]>();
  for (const qid of afterTestLeakage) {
    const key = normalizeSciFactQuery(queryById.get(qid)!);
    const group = trainByNormalized.get(key) ?? [];
    group.push(qid);
    trainByNormalized.set(key, group);
  }
  const safeQids: string[] = [];
  for (const group of trainByNormalized.values()) {
    group.sort(byteCompare);
    safeQids.push(group[0]!);
    for (const qid of group.slice(1)) {
      exclusions.push({ qid, reason: "duplicate_train_query", duplicate_of_qid: group[0]! });
    }
  }
  exclusions.sort((left, right) => byteCompare(left.qid, right.qid));

  const ranked = safeQids.sort((left, right) => {
    const keyOrder = byteCompare(sciFactSplitKey(left), sciFactSplitKey(right));
    return keyOrder || byteCompare(left, right);
  });
  if (ranked.length <= SCIFACT_DISTILL_VALIDATION_COUNT) {
    throw new Error(
      `safe train pool has ${ranked.length} queries; need more than ${SCIFACT_DISTILL_VALIDATION_COUNT}`,
    );
  }
  const valQids = ranked.slice(0, SCIFACT_DISTILL_VALIDATION_COUNT).sort(byteCompare);
  const finalTrainQids = ranked.slice(SCIFACT_DISTILL_VALIDATION_COUNT).sort(byteCompare);
  return {
    counts: {
      source_train_queries: trainQids.size,
      source_test_queries: testQids.size,
      excluded_queries: exclusions.length,
      train_queries: finalTrainQids.length,
      val_queries: valQids.length,
    },
    exclusions,
    train_qids: finalTrainQids,
    val_qids: valQids,
  };
}

export function compareCanonicalMetrics(
  left: CanonicalQueryMetrics,
  right: CanonicalQueryMetrics,
): number {
  for (const key of ["recall_at_30", "mrr_at_10", "ndcg_at_10"] as const) {
    const delta = left[key]! - right[key]!;
    if (delta !== 0) return delta > 0 ? 1 : -1;
  }
  return 0;
}

/**
 * A query has headroom when its unexpanded retrieval is not already perfect.
 * Winners can only come from these queries: canonical metrics are bounded by
 * 1, so nothing can rank strictly above a raw (1, 1, 1). Any winner threshold
 * above the headroom count is therefore unsatisfiable by any teacher.
 */
export function hasRetrievalHeadroom(record: SciFactDistillRecord): boolean {
  return !!record.raw_metrics && compareCanonicalMetrics(
    { recall_at_30: 1, mrr_at_10: 1, ndcg_at_10: 1 },
    record.raw_metrics,
  ) > 0;
}

export function selectDistillWinner(
  candidates: readonly DistillCandidate[],
  rawMetrics: CanonicalQueryMetrics,
): Pick<SciFactDistillRecord, "selected_candidate_index" | "selection_status"> {
  const valid = candidates.filter(candidate => candidateIsAdmitted(candidate) && candidate.metrics);
  if (valid.length === 0) {
    return { selected_candidate_index: null, selection_status: "no_valid_candidate" };
  }
  const best = valid.reduce((current, candidate) => {
    const scoreOrder = compareCanonicalMetrics(candidate.metrics!, current.metrics!);
    if (scoreOrder > 0) return candidate;
    if (scoreOrder === 0 && candidate.candidate_index < current.candidate_index) return candidate;
    return current;
  });
  if (compareCanonicalMetrics(best.metrics!, rawMetrics) <= 0) {
    return { selected_candidate_index: null, selection_status: "no_winner" };
  }
  return { selected_candidate_index: best.candidate_index, selection_status: "winner" };
}

export function candidateIsAdmitted(candidate: DistillCandidate): boolean {
  return candidate.contract?.valid === true && candidate.semantic_gate?.valid !== false;
}

/**
 * A resumable partial artifact must be an exact qid-prefix of the expected row
 * order; anything else means the partial file belongs to a different run.
 */
export function assertQidPrefix(
  label: string,
  partial: readonly { qid: string }[],
  expected: readonly { qid: string }[],
): void {
  if (partial.length > expected.length) throw new Error(`${label} has extra rows`);
  for (let index = 0; index < partial.length; index++) {
    if (partial[index]!.qid !== expected[index]!.qid) {
      throw new Error(`${label} is not a prefix at row ${index + 1}`);
    }
  }
}

export function validateDistillRecordShape(record: SciFactDistillRecord): void {
  if (!record.qid || !record.query) throw new Error("distill record needs qid and query");
  if (record.split !== "train" && record.split !== "val") {
    throw new Error(`qid ${record.qid}: split must be train or val`);
  }
  if (record.candidates.length !== SCIFACT_DISTILL_CANDIDATE_COUNT) {
    throw new Error(
      `qid ${record.qid}: expected ${SCIFACT_DISTILL_CANDIDATE_COUNT} candidates, got ${record.candidates.length}`,
    );
  }
  for (let index = 0; index < record.candidates.length; index++) {
    if (record.candidates[index]!.candidate_index !== index) {
      throw new Error(`qid ${record.qid}: candidate indexes must be contiguous from zero`);
    }
  }
}

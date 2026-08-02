/**
 * Types for the QMD benchmark harness.
 *
 * A benchmark fixture defines queries with expected results.
 * The harness runs each query through multiple search backends
 * and measures precision, recall, MRR, and latency.
 */

export interface BenchmarkQuery {
  /** Unique identifier for the query */
  id: string;
  /** The search query text */
  query: string;
  /** Query difficulty/type for grouping results */
  type: "exact" | "semantic" | "topical" | "cross-domain" | "alias";
  /** Human-readable description of what this tests */
  description: string;
  /** File paths (relative to collection) that should appear in results */
  expected_files: string[];
  /** How many of expected_files should appear in top-k results */
  expected_in_top_k: number;
}

export interface BenchmarkFixture {
  /** Description of the benchmark */
  description: string;
  /** Fixture format version */
  version: number;
  /** Optional collection to search within */
  collection?: string;
  /** The test queries */
  queries: BenchmarkQuery[];
}

export interface BackendResult {
  /** Fraction of top-k results that are relevant */
  precision_at_k: number;
  /** Fraction of expected files found anywhere in results */
  recall: number;
  /** Fraction of expected files found in the first result */
  recall_at_1: number;
  /** Fraction of expected files found in the top 3 results */
  recall_at_3: number;
  /** Fraction of expected files found in the top 5 results */
  recall_at_5: number;
  /** Reciprocal rank of first relevant result (1/rank, 0 if not found) */
  mrr: number;
  /** Harmonic mean of precision_at_k and recall */
  f1: number;
  /** Number of expected files found in top-k */
  hits_at_k: number;
  /** Total expected files */
  total_expected: number;
  /** Wall-clock latency in milliseconds */
  latency_ms: number;
  /** Top result file paths (for inspection) */
  top_files: string[];
  /** Expected files that were found anywhere in the returned result set */
  matched_files: string[];
  /** Expected files missing from the returned result set */
  unmatched_expected_files: string[];
}

export interface QueryResult {
  id: string;
  query: string;
  type: string;
  backends: Record<string, BackendResult>;
}

export interface BenchmarkResult {
  timestamp: string;
  fixture: string;
  results: QueryResult[];
  summary: Record<string, {
    avg_precision: number;
    avg_recall: number;
    avg_recall_at_1: number;
    avg_recall_at_3: number;
    avg_recall_at_5: number;
    avg_mrr: number;
    avg_f1: number;
    avg_latency_ms: number;
  }>;
}

// ---------------------------------------------------------------------------
// Canonical qrels benchmark (v2)
// ---------------------------------------------------------------------------

export const QMD_EXPANSION_SCIFACT_BENCHMARK_ID = "qmd-expansion-scifact-v1";

export type CanonicalMetric =
  | "recall_at_cutoffs"
  | "mrr_at_10"
  | "ndcg_at_10";

export type ExpansionType = "lex" | "vec" | "hyde";
export type ExpansionStatus = "ok" | "format_error" | "generation_error";
export type BenchmarkVariant = "raw" | "current" | "candidate";
export type RunStatus = "completed" | "failed";
export type QueryExecutionStatus = "ok" | "error";

export interface BenchmarkManifestV2 {
  benchmark_id: string;
  source: {
    url: string;
    archive_md5: string;
    split: string;
  };
  source_qrels_sha256: string;
  excluded_qids_sha256: string;
  leakage_report_sha256: string;
  converted_data_sha256: string;
  qrels: {
    relevant_threshold: number;
    unjudged: "nonrelevant";
    graded: false;
  };
  cutoffs: number[];
  metrics: CanonicalMetric[];
}

export interface BenchmarkQueryV2 {
  qid: string;
  query: string;
}

export interface BenchmarkQrel {
  qid: string;
  doc_id: string;
  relevance: 0 | 1;
}

export interface BenchmarkDocument {
  doc_id: string;
  path: string;
}

export type BenchmarkExpansion = [ExpansionType, string];

export interface BenchmarkExpansionRecord {
  qid: string;
  query: string;
  status: ExpansionStatus;
  raw_output: string;
  output: BenchmarkExpansion[];
  fallback_used: boolean;
  error: string | null;
}

export interface RetrievalProfileV2 {
  profile_id: string;
  collection_name: string;
  collection_root: string;
  embedding_model: string;
  reranker_model: string | null;
  result_limit: number;
  per_list_limit: number;
  candidate_limit: number;
  rerank: boolean;
  auto_expand: false;
  strong_signal_bypass: false;
}

export interface IndexManifestV2 {
  collection_name: string;
  collection_root: string;
  documents_sha256: string;
  embedding_model: string;
  embedding_fingerprint: string;
  document_count: number;
  vector_document_count: number;
  vector_chunk_count: number;
  pending_embedding_count: number;
  index_fingerprint: string;
}

export interface CanonicalQueryMetrics {
  [metric: `recall_at_${number}`]: number;
  mrr_at_10: number;
  ndcg_at_10: number;
}

export interface BenchmarkRunMetrics extends CanonicalQueryMetrics {
  expansion_pass_rate: number | null;
  format_error_rate: number | null;
  generation_error_rate: number | null;
  fallback_rate: number | null;
}

export interface CanonicalRankingEntry {
  rank: number;
  doc_id: string;
  relevance: 0 | 1 | null;
}

export interface CanonicalQueryDiagnostic {
  code: "ranking_below_result_limit";
  expected: number;
  actual: number;
}

export interface CanonicalQueryResult {
  qid: string;
  variant: BenchmarkVariant;
  retrieval_status: QueryExecutionStatus;
  expansion_status: ExpansionStatus | null;
  expansions: BenchmarkExpansion[];
  ranking: CanonicalRankingEntry[];
  latency_ms: number;
  metrics: CanonicalQueryMetrics | null;
  fallback_used: boolean | null;
  expansion_error: string | null;
  retrieval_error: string | null;
  diagnostics: CanonicalQueryDiagnostic[];
}

export interface BenchmarkRunV2 {
  run_id: string;
  benchmark_id: string;
  benchmark_manifest_sha256: string;
  retrieval_profile: string;
  retrieval_profile_sha256: string;
  qmd_commit: string;
  qmd_dirty: boolean;
  qmd_diff_sha256: string | null;
  qmd_config_sha256: string;
  collection_name: string;
  collection_root: string;
  index_manifest_sha256: string;
  index_fingerprint: string;
  embedding_artifact_sha256: string;
  reranker_artifact_sha256: string | null;
  variant: BenchmarkVariant;
  expansion_model: string | null;
  expansions_sha256: string | null;
  retrieval: {
    result_limit: number;
    per_list_limit: number;
    candidate_limit: number;
  };
  command: string[];
  runtime: {
    qmd: string;
    bun_or_node: string;
    sqlite: string;
    sqlite_vec: string;
    platform: string;
  };
  status: RunStatus;
  results: string;
  metrics: BenchmarkRunMetrics | null;
  expansion_failures: {
    expansion_pass_count: number;
    format_error_count: number;
    generation_error_count: number;
    fallback_count: number;
    expansion_pass_rate: number | null;
    format_error_rate: number | null;
    generation_error_rate: number | null;
    fallback_rate: number | null;
  };
}

export interface LoadedBenchmarkV2 {
  manifest: BenchmarkManifestV2;
  queries: BenchmarkQueryV2[];
  qrels: BenchmarkQrel[];
  documents: BenchmarkDocument[];
  queryById: Map<string, BenchmarkQueryV2>;
  documentById: Map<string, BenchmarkDocument>;
  relevanceByQuery: Map<string, Map<string, 0 | 1>>;
}

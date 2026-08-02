/**
 * Scoring functions for the QMD benchmark harness.
 *
 * Computes precision@k, recall, MRR, and F1 for search results
 * against ground-truth expected files.
 */

/**
 * Normalize a file path for comparison.
 * Strips qmd:// prefix, lowercases, removes leading/trailing slashes.
 */
export function normalizePath(p: string): string {
  if (p.startsWith("qmd://")) {
    // qmd://collection/docs/readme.md → docs/readme.md
    const withoutScheme = p.slice("qmd://".length);
    const slashIdx = withoutScheme.indexOf("/");
    p = slashIdx >= 0 ? withoutScheme.slice(slashIdx + 1) : withoutScheme;
  }
  return p.toLowerCase().replace(/^\/+|\/+$/g, "");
}

/**
 * Check if two paths refer to the same file.
 * Handles different path formats by comparing normalized suffixes.
 */
export function pathsMatch(result: string, expected: string): boolean {
  const nr = normalizePath(result);
  const ne = normalizePath(expected);
  if (nr === ne) return true;
  if (nr.endsWith(ne) || ne.endsWith(nr)) return true;
  return false;
}

type ScoreMetrics = {
  precision_at_k: number;
  recall: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  mrr: number;
  f1: number;
  hits_at_k: number;
  matched_files: string[];
  unmatched_expected_files: string[];
};

function hitsWithin(resultFiles: string[], expectedFiles: string[], k: number): number {
  const topKResults = resultFiles.slice(0, k);
  let hits = 0;
  for (const expected of expectedFiles) {
    if (topKResults.some(r => pathsMatch(r, expected))) {
      hits++;
    }
  }
  return hits;
}

/**
 * Score a set of search results against expected files.
 */
export function scoreResults(
  resultFiles: string[],
  expectedFiles: string[],
  topK: number,
): ScoreMetrics {
  // Count hits in top-k
  const hitsAtK = hitsWithin(resultFiles, expectedFiles, topK);

  const matchedFiles: string[] = [];
  const unmatchedExpectedFiles: string[] = [];

  for (const expected of expectedFiles) {
    if (resultFiles.some(r => pathsMatch(r, expected))) {
      matchedFiles.push(expected);
    } else {
      unmatchedExpectedFiles.push(expected);
    }
  }

  // MRR: reciprocal rank of first relevant result
  let mrr = 0;
  for (let i = 0; i < resultFiles.length; i++) {
    if (expectedFiles.some(e => pathsMatch(resultFiles[i]!, e))) {
      mrr = 1 / (i + 1);
      break;
    }
  }

  const denominator = Math.min(topK, expectedFiles.length);
  const precision_at_k = denominator > 0 ? hitsAtK / denominator : 0;
  const recall = expectedFiles.length > 0 ? matchedFiles.length / expectedFiles.length : 0;
  const recall_at_1 = expectedFiles.length > 0 ? hitsWithin(resultFiles, expectedFiles, 1) / expectedFiles.length : 0;
  const recall_at_3 = expectedFiles.length > 0 ? hitsWithin(resultFiles, expectedFiles, 3) / expectedFiles.length : 0;
  const recall_at_5 = expectedFiles.length > 0 ? hitsWithin(resultFiles, expectedFiles, 5) / expectedFiles.length : 0;
  const f1 = precision_at_k + recall > 0
    ? 2 * (precision_at_k * recall) / (precision_at_k + recall)
    : 0;

  return {
    precision_at_k,
    recall,
    recall_at_1,
    recall_at_3,
    recall_at_5,
    mrr,
    f1,
    hits_at_k: hitsAtK,
    matched_files: matchedFiles,
    unmatched_expected_files: unmatchedExpectedFiles,
  };
}

export type CanonicalMetrics = {
  [metric: `recall_at_${number}`]: number;
  mrr_at_10: number;
  ndcg_at_10: number;
};

/**
 * Score one source-document ranking with standard binary qrels metrics.
 * The caller must aggregate QMD chunk hits to unique source documents first.
 */
export function scoreCanonicalRanking(
  rankedDocIds: string[],
  relevantDocIds: ReadonlySet<string>,
  cutoffs: readonly number[],
): CanonicalMetrics {
  if (relevantDocIds.size === 0) {
    throw new Error("Cannot score a query with no relevant documents");
  }

  const seenRanked = new Set<string>();
  for (const docId of rankedDocIds) {
    if (seenRanked.has(docId)) {
      throw new Error(`Ranking contains duplicate doc_id "${docId}"`);
    }
    seenRanked.add(docId);
  }

  const metrics = {} as CanonicalMetrics;
  for (const cutoff of cutoffs) {
    if (!Number.isSafeInteger(cutoff) || cutoff <= 0) {
      throw new Error(`Invalid cutoff ${JSON.stringify(cutoff)}`);
    }
    let relevantFound = 0;
    for (const docId of rankedDocIds.slice(0, cutoff)) {
      if (relevantDocIds.has(docId)) relevantFound++;
    }
    metrics[`recall_at_${cutoff}`] = relevantFound / relevantDocIds.size;
  }

  const firstRelevantIndex = rankedDocIds
    .slice(0, 10)
    .findIndex(docId => relevantDocIds.has(docId));
  metrics.mrr_at_10 = firstRelevantIndex === -1 ? 0 : 1 / (firstRelevantIndex + 1);

  let dcg = 0;
  for (let index = 0; index < Math.min(10, rankedDocIds.length); index++) {
    if (relevantDocIds.has(rankedDocIds[index]!)) {
      dcg += 1 / Math.log2(index + 2);
    }
  }
  let idcg = 0;
  for (let index = 0; index < Math.min(10, relevantDocIds.size); index++) {
    idcg += 1 / Math.log2(index + 2);
  }
  metrics.ndcg_at_10 = dcg / idcg;
  return metrics;
}

/** Macro-average query metrics. Execution errors must be excluded by the caller. */
export function averageCanonicalMetrics(
  queryMetrics: readonly CanonicalMetrics[],
  cutoffs: readonly number[],
): CanonicalMetrics {
  if (queryMetrics.length === 0) {
    throw new Error("Cannot summarize zero successful queries");
  }
  const summary = {} as CanonicalMetrics;
  for (const cutoff of cutoffs) {
    const key = `recall_at_${cutoff}` as const;
    summary[key] = queryMetrics.reduce((sum, metrics) => sum + metrics[key]!, 0)
      / queryMetrics.length;
  }
  summary.mrr_at_10 = queryMetrics.reduce((sum, metrics) => sum + metrics.mrr_at_10, 0)
    / queryMetrics.length;
  summary.ndcg_at_10 = queryMetrics.reduce((sum, metrics) => sum + metrics.ndcg_at_10, 0)
    / queryMetrics.length;
  return summary;
}

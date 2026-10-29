import { createHash } from "node:crypto";
import type { CanonicalQueryMetrics } from "../../../src/bench/types.js";

export type PublicSelectionStatus =
  | "winner"
  | "qualified_tie"
  | "no_winner"
  | "no_valid_candidate";

export function comparePublicMetrics(
  left: CanonicalQueryMetrics,
  right: CanonicalQueryMetrics,
): number {
  for (const key of ["recall_at_10", "ndcg_at_10", "mrr_at_10"] as const) {
    if (left[key] !== right[key]) return left[key] > right[key] ? 1 : -1;
  }
  return 0;
}

export function publicTieIndex(sourceId: string, qid: string, tiedCount: number): number {
  if (!Number.isSafeInteger(tiedCount) || tiedCount <= 0) throw new Error("tiedCount must be positive");
  const digest = createHash("sha256")
    .update(`qmd-public-v0\0seed=42\0${sourceId}\0${qid}`)
    .digest("hex");
  return Number(BigInt(`0x${digest}`) % BigInt(tiedCount));
}

/**
 * Strict metric-order selection (nf-freshstack): the best candidate under
 * `order` must beat raw; equal metrics resolve to the lower candidate_index,
 * which `order` must state as its final tie-breaker. No qualified ties.
 */
export function selectByMetricOrder(options: {
  order: readonly string[];
  raw: CanonicalQueryMetrics;
  candidates: { candidate_index: number; metrics: CanonicalQueryMetrics }[];
}): { selection_status: PublicSelectionStatus; selected_candidate_index: number | null } {
  const metrics = options.order.slice(0, -1) as (keyof CanonicalQueryMetrics)[];
  if (options.order.at(-1) !== "lower_candidate_index" || metrics.length === 0) {
    throw new Error(`winner order must list metrics then "lower_candidate_index": ${options.order.join(",")}`);
  }
  const compare = (left: CanonicalQueryMetrics, right: CanonicalQueryMetrics): number => {
    for (const key of metrics) {
      const a = left[key];
      const b = right[key];
      if (a === undefined || b === undefined) throw new Error(`missing metric ${key}`);
      if (a !== b) return a > b ? 1 : -1;
    }
    return 0;
  };
  if (options.candidates.length === 0) {
    return { selection_status: "no_valid_candidate", selected_candidate_index: null };
  }
  const ordered = [...options.candidates].sort((left, right) => left.candidate_index - right.candidate_index);
  let selected = ordered[0]!;
  for (const candidate of ordered.slice(1)) {
    if (compare(candidate.metrics, selected.metrics) > 0) selected = candidate;
  }
  return compare(selected.metrics, options.raw) > 0
    ? { selection_status: "winner", selected_candidate_index: selected.candidate_index }
    : { selection_status: "no_winner", selected_candidate_index: null };
}

export function selectPublicCandidate(options: {
  sourceId: string;
  qid: string;
  raw: CanonicalQueryMetrics;
  candidates: { candidate_index: number; metrics: CanonicalQueryMetrics }[];
}): { selection_status: PublicSelectionStatus; selected_candidate_index: number | null } {
  if (options.candidates.length === 0) {
    return { selection_status: "no_valid_candidate", selected_candidate_index: null };
  }
  const ordered = [...options.candidates].sort((left, right) => left.candidate_index - right.candidate_index);
  let best = ordered[0]!.metrics;
  for (const candidate of ordered.slice(1)) {
    if (comparePublicMetrics(candidate.metrics, best) > 0) best = candidate.metrics;
  }
  const tied = ordered.filter(candidate => comparePublicMetrics(candidate.metrics, best) === 0);
  const selected = tied[publicTieIndex(options.sourceId, options.qid, tied.length)]!;
  const rawComparison = comparePublicMetrics(best, options.raw);
  return {
    selection_status: rawComparison > 0
      ? "winner"
      : rawComparison === 0 ? "qualified_tie" : "no_winner",
    selected_candidate_index: rawComparison >= 0 ? selected.candidate_index : null,
  };
}

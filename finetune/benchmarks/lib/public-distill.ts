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

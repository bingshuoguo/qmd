import { describe, expect, test } from "vitest";
import type { CanonicalQueryMetrics } from "../src/bench/types.js";
import {
  assertQidPrefix,
  buildSciFactDistillMembership,
  candidateIsAdmitted,
  compareCanonicalMetrics,
  hasRetrievalHeadroom,
  normalizeSciFactQuery,
  sciFactSplitKey,
  selectDistillWinner,
  type DistillCandidate,
  type SciFactDistillRecord,
  type SciFactQrel,
} from "../finetune/benchmarks/lib/distill.js";

function qrel(qid: string): SciFactQrel {
  return { qid, doc_id: `doc-${qid}`, relevance: 1 };
}

function metrics(recall: number, mrr: number, ndcg: number): CanonicalQueryMetrics {
  return { recall_at_30: recall, mrr_at_10: mrr, ndcg_at_10: ndcg };
}

function candidate(
  candidateIndex: number,
  candidateMetrics: CanonicalQueryMetrics | null,
  valid = true,
): DistillCandidate {
  return {
    candidate_index: candidateIndex,
    generation_status: "ok",
    raw_output: "lex: alpha\nvec: evidence about alpha",
    parsed_output: [["lex", "alpha"], ["vec", "evidence about alpha"]],
    generation_error: null,
    contract: {
      version: "training-target-v1.1",
      valid,
      errors: [],
      warnings: [],
      canonical_output: [["lex", "alpha"], ["vec", "evidence about alpha"]],
    },
    metrics: candidateMetrics,
  };
}

describe("SciFact distillation split", () => {
  test("accepts a deterministic prefix and rejects reordered or oversized pilot records", () => {
    const expected = [{ qid: "q1" }, { qid: "q2" }, { qid: "q3" }];
    expect(() => assertQidPrefix("pilot", expected.slice(0, 2), expected)).not.toThrow();
    expect(() => assertQidPrefix("pilot", [{ qid: "q2" }], expected)).toThrow("not a prefix");
    expect(() => assertQidPrefix("pilot", [...expected, { qid: "q4" }], expected))
      .toThrow("extra rows");
  });

  test("normalizes exact query identity without semantic clustering", () => {
    expect(normalizeSciFactQuery("  ＡLPHA\t Evidence  ")).toBe("alpha evidence");
  });

  test("excludes exact test leakage and duplicate train text before deterministic 161-val split", () => {
    const base = Array.from({ length: 166 }, (_, index) => ({
      qid: `q${index.toString().padStart(3, "0")}`,
      query: `unique query ${index}`,
    }));
    const queries = [
      ...base,
      { qid: "q-leak", query: "same as test" },
      { qid: "q999", query: " UNIQUE   QUERY 0 " },
      { qid: "t1", query: "Same As Test" },
      { qid: "t2", query: "held out" },
    ];
    const trainQrels = [...base.map(query => qrel(query.qid)), qrel("q-leak"), qrel("q999")];
    const result = buildSciFactDistillMembership(
      queries,
      trainQrels,
      [qrel("t1"), qrel("t2")],
    );

    expect(result.counts).toEqual({
      source_train_queries: 168,
      source_test_queries: 2,
      excluded_queries: 2,
      train_queries: 5,
      val_queries: 161,
    });
    expect(result.exclusions).toEqual([
      { qid: "q-leak", reason: "exact_test_query_overlap", duplicate_of_qid: "t1" },
      { qid: "q999", reason: "duplicate_train_query", duplicate_of_qid: "q000" },
    ]);
    expect(new Set([...result.train_qids, ...result.val_qids]).size).toBe(166);
    expect(result.val_qids).toEqual(
      [...base]
        .sort((left, right) => sciFactSplitKey(left.qid).localeCompare(sciFactSplitKey(right.qid)))
        .slice(0, 161)
        .map(query => query.qid)
        .sort(),
    );
  });
});

describe("SciFact distillation winner rule", () => {
  test("compares Recall@30, then MRR@10, then nDCG@10", () => {
    expect(compareCanonicalMetrics(metrics(1, 0, 0), metrics(0.5, 1, 1))).toBe(1);
    expect(compareCanonicalMetrics(metrics(1, 0.5, 0), metrics(1, 0.25, 1))).toBe(1);
    expect(compareCanonicalMetrics(metrics(1, 0.5, 0.8), metrics(1, 0.5, 0.7))).toBe(1);
  });

  test("selects the smallest candidate index on an improving score tie", () => {
    const tied = metrics(1, 0.5, 0.8);
    expect(selectDistillWinner(
      [candidate(1, tied), candidate(0, tied)],
      metrics(0.5, 1, 1),
    )).toEqual({ selected_candidate_index: 0, selection_status: "winner" });
  });

  test("does not force a candidate that only ties raw", () => {
    const raw = metrics(1, 0.5, 0.8);
    expect(selectDistillWinner([candidate(0, raw)], raw)).toEqual({
      selected_candidate_index: null,
      selection_status: "no_winner",
    });
  });

  test("reports no valid candidate when Contract rejects every candidate", () => {
    expect(selectDistillWinner([candidate(0, metrics(1, 1, 1), false)], metrics(0, 0, 0)))
      .toEqual({ selected_candidate_index: null, selection_status: "no_valid_candidate" });
  });

  // The SciFact gate no longer emits any blocking error, but the selection
  // mechanism must still honour one so a future gate can be wired in.
  test("excludes a candidate rejected by an optional semantic gate", () => {
    const rejected = candidate(0, metrics(1, 1, 1));
    rejected.semantic_gate = {
      version: "future-gate-v1",
      valid: false,
      errors: [{ code: "some_blocking_check", path: "output[1]", message: "blocked" }],
    };
    expect(candidateIsAdmitted(rejected)).toBe(false);
    expect(selectDistillWinner([rejected], metrics(0, 0, 0))).toEqual({
      selected_candidate_index: null,
      selection_status: "no_valid_candidate",
    });
  });

  test("admits a candidate carrying only semantic gate advisories", () => {
    const advised = candidate(0, metrics(1, 1, 1));
    advised.semantic_gate = {
      version: "scifact-observational-v3",
      valid: true,
      errors: [],
      advisories: [{ code: "negation_lost", path: "output[1]", message: "negation" }],
    };
    expect(candidateIsAdmitted(advised)).toBe(true);
    expect(selectDistillWinner([advised], metrics(0, 0, 0))).toEqual({
      selected_candidate_index: 0,
      selection_status: "winner",
    });
  });

  test("counts a query as headroom only when raw retrieval is not already perfect", () => {
    const record = (raw: CanonicalQueryMetrics | null): SciFactDistillRecord => ({
      qid: "q1",
      split: "train",
      query: "alpha",
      raw_metrics: raw,
      candidates: [],
      selected_candidate_index: null,
      selection_status: "pending",
    });
    expect(hasRetrievalHeadroom(record(metrics(1, 1, 1)))).toBe(false);
    expect(hasRetrievalHeadroom(record(metrics(1, 1, 0.999)))).toBe(true);
    expect(hasRetrievalHeadroom(record(metrics(0.9, 0.5, 0.5)))).toBe(true);
    expect(hasRetrievalHeadroom(record(null))).toBe(false);
  });

  test("no candidate can win a query whose raw retrieval is already perfect", () => {
    // This is what makes a winner threshold above the headroom count
    // unsatisfiable: canonical metrics are capped at 1.
    expect(selectDistillWinner([candidate(0, metrics(1, 1, 1))], metrics(1, 1, 1)))
      .toEqual({ selected_candidate_index: null, selection_status: "no_winner" });
  });
});

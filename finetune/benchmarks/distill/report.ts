#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import type { CanonicalQueryMetrics } from "../../../src/bench/types.js";
import {
  candidateIsAdmitted,
  compareCanonicalMetrics,
  hasRetrievalHeadroom,
  parseDistillRecords,
  sha256,
  type SciFactDistillRecord,
} from "../lib/distill.js";

type SemanticSeverity = "safe" | "moderate" | "critical";
type SemanticAudit = {
  version: "scifact-distill-semantic-audit-v1";
  entries: {
    qid: string;
    candidate_index: number;
    severity: SemanticSeverity;
    notes?: string;
  }[];
};

type RunReport = ReturnType<typeof buildReport>;

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run distill:report -- --run-dir <dir> [--semantic-audit <json>] \\",
    "    [--baseline-run-dir <v1-clean-dir>] [--baseline-semantic-audit <json>]",
  ].join("\n");
  (exitCode === 0 ? console.log : console.error)(message);
  process.exit(exitCode);
}

function loadAudit(path: string | undefined): SemanticAudit | null {
  if (!path) return null;
  const audit = JSON.parse(readFileSync(resolve(path), "utf8")) as SemanticAudit;
  if (audit.version !== "scifact-distill-semantic-audit-v1" || !Array.isArray(audit.entries)) {
    throw new Error("Unsupported semantic audit format");
  }
  const seen = new Set<string>();
  for (const entry of audit.entries) {
    if (
      !entry.qid
      || !Number.isSafeInteger(entry.candidate_index)
      || !["safe", "moderate", "critical"].includes(entry.severity)
    ) {
      throw new Error("Semantic audit contains an invalid entry");
    }
    const key = `${entry.qid}\0${entry.candidate_index}`;
    if (seen.has(key)) throw new Error(`Semantic audit contains duplicate entry ${entry.qid}`);
    seen.add(key);
  }
  return audit;
}

function average(metrics: readonly CanonicalQueryMetrics[]): CanonicalQueryMetrics | null {
  if (metrics.length === 0) return null;
  return {
    recall_at_30: metrics.reduce((sum, item) => sum + item.recall_at_30, 0) / metrics.length,
    mrr_at_10: metrics.reduce((sum, item) => sum + item.mrr_at_10, 0) / metrics.length,
    ndcg_at_10: metrics.reduce((sum, item) => sum + item.ndcg_at_10, 0) / metrics.length,
  };
}

function selectedMetrics(record: SciFactDistillRecord): CanonicalQueryMetrics | null {
  if (record.selected_candidate_index === null) return record.raw_metrics;
  return record.candidates[record.selected_candidate_index]?.metrics ?? null;
}

function buildReport(
  runDir: string,
  records: readonly SciFactDistillRecord[],
  candidatesBytes: Buffer,
  audit: SemanticAudit | null,
) {
  const manifest = JSON.parse(readFileSync(join(runDir, "manifest.json"), "utf8")) as Record<string, unknown>;
  const candidates = records.flatMap(record => record.candidates);
  const generated = candidates.filter(candidate => candidate.generation_status !== "generation_error");
  const generationErrors = candidates.length - generated.length;
  const errorTypeCounts: Record<string, number> = {};
  for (const candidate of candidates) {
    if (candidate.generation_status !== "generation_error") continue;
    const type = candidate.generation_error_type ?? "legacy_unclassified";
    errorTypeCounts[type] = (errorTypeCounts[type] ?? 0) + 1;
  }
  const coveredQueries = records.filter(record => (
    record.candidates.some(candidate => candidate.generation_status !== "generation_error")
  )).length;
  const contractValid = generated.filter(candidate => candidate.contract?.valid).length;
  // The generator writes `semantic_gate: null` when no gate was requested, so
  // `!== undefined` counted an absent gate as present and reported a 0% pass
  // rate for runs that never ran one.
  const semanticGatePresent = generated.filter(candidate => candidate.semantic_gate != null).length;
  const semanticGateValid = generated.filter(candidate => candidate.semantic_gate?.valid === true).length;
  const semanticAdmittedQueries = records.filter(record => (
    record.candidates.some(candidateIsAdmitted)
  )).length;
  const expansionCounts = generated.reduce((counts, candidate) => {
    for (const [type] of candidate.parsed_output) counts[type]++;
    return counts;
  }, { lex: 0, vec: 0, hyde: 0 });
  const comparisons = { better: 0, tie: 0, worse: 0 };
  for (const record of records) {
    if (!record.raw_metrics) continue;
    for (const candidate of record.candidates) {
      if (!candidateIsAdmitted(candidate) || !candidate.metrics) continue;
      const order = compareCanonicalMetrics(candidate.metrics, record.raw_metrics);
      comparisons[order > 0 ? "better" : order < 0 ? "worse" : "tie"]++;
    }
  }
  const compared = comparisons.better + comparisons.tie + comparisons.worse;
  const headroom = records.filter(hasRetrievalHeadroom);
  const headroomCovered = headroom.filter(record => (
    record.candidates.some(candidate => candidate.generation_status !== "generation_error")
  )).length;
  const winners = headroom.filter(record => record.selection_status === "winner");
  const auditByWinner = new Map(
    (audit?.entries ?? []).map(entry => [`${entry.qid}\0${entry.candidate_index}`, entry]),
  );
  const winnerAudits = winners.map(record => (
    auditByWinner.get(`${record.qid}\0${record.selected_candidate_index}`) ?? null
  ));
  const severityCounts = { safe: 0, moderate: 0, critical: 0 };
  for (const entry of winnerAudits) if (entry) severityCounts[entry.severity]++;
  const missingWinnerAudits = winners
    .filter((_, index) => winnerAudits[index] === null)
    .map(record => ({ qid: record.qid, candidate_index: record.selected_candidate_index }));
  return {
    version: "scifact-distill-report-v1",
    generated_at: new Date().toISOString(),
    run_dir: runDir,
    candidates_sha256: sha256(candidatesBytes),
    experiment: {
      id: manifest.experiment_id ?? basename(runDir),
      teacher_model: manifest.teacher_model ?? null,
      prompt_version: manifest.prompt_version ?? null,
      prompt_sha256: manifest.prompt_sha256 ?? null,
      max_output_tokens: manifest.max_output_tokens ?? 1200,
      reasoning_effort: manifest.reasoning_effort ?? null,
    },
    generation: {
      queries: records.length,
      candidates: candidates.length,
      errors: generationErrors,
      error_type_counts: errorTypeCounts,
      error_rate: candidates.length === 0 ? null : generationErrors / candidates.length,
      covered_queries: coveredQueries,
      query_coverage_rate: records.length === 0 ? null : coveredQueries / records.length,
      contract_valid: contractValid,
      contract_pass_rate: generated.length === 0 ? null : contractValid / generated.length,
      semantic_gate_present: semanticGatePresent,
      semantic_gate_valid: semanticGateValid,
      semantic_gate_pass_rate: semanticGatePresent === 0 ? null : semanticGateValid / semanticGatePresent,
      semantic_admitted_queries: semanticGatePresent === 0 ? null : semanticAdmittedQueries,
      semantic_query_coverage_rate: semanticGatePresent === 0
        ? null
        : semanticAdmittedQueries / records.length,
      average_expansions: generated.length === 0 ? null : {
        lex: expansionCounts.lex / generated.length,
        vec: expansionCounts.vec / generated.length,
        hyde: expansionCounts.hyde / generated.length,
      },
    },
    retrieval: {
      headroom_queries: headroom.length,
      headroom_qids: headroom.map(record => record.qid),
      headroom_covered_queries: headroomCovered,
      headroom_query_coverage_rate: headroom.length === 0 ? null : headroomCovered / headroom.length,
      headroom_semantic_admitted_queries: semanticGatePresent === 0
        ? null
        : headroom.filter(record => record.candidates.some(candidateIsAdmitted)).length,
      headroom_semantic_query_coverage_rate: semanticGatePresent === 0 || headroom.length === 0
        ? null
        : headroom.filter(record => record.candidates.some(candidateIsAdmitted)).length / headroom.length,
      winners: winners.length,
      headroom_conversion: headroom.length === 0 ? null : winners.length / headroom.length,
      candidate_comparisons: comparisons,
      candidate_worse_rate: compared === 0 ? null : comparisons.worse / compared,
      selected_or_raw_macro: average(records.map(selectedMetrics).filter(metric => metric !== null)),
      headroom_selected_or_raw_macro: average(headroom.map(selectedMetrics).filter(metric => metric !== null)),
    },
    semantic: audit === null ? null : {
      audit_version: audit.version,
      winner_counts: severityCounts,
      semantic_admitted_winners: severityCounts.safe,
      semantic_admitted_conversion: headroom.length === 0 ? null : severityCounts.safe / headroom.length,
      critical_rate: winners.length === 0 ? 0 : severityCounts.critical / winners.length,
      complete: missingWinnerAudits.length === 0,
      missing_winner_audits: missingWinnerAudits,
    },
  };
}

function compareRuns(baseline: RunReport, candidate: RunReport) {
  const baselineMetrics = baseline.retrieval.headroom_selected_or_raw_macro;
  const candidateMetrics = candidate.retrieval.headroom_selected_or_raw_macro;
  const winnerDelta = candidate.retrieval.winners - baseline.retrieval.winners;
  const conversionDelta = (candidate.retrieval.headroom_conversion ?? 0)
    - (baseline.retrieval.headroom_conversion ?? 0);
  // Both runs are asserted to share the same headroom qids, so
  // conversionDelta === winnerDelta / headroom. A separate conversion
  // threshold would restate winner_delta_at_least_4 rather than add to it.
  const checks: Record<string, boolean> = {
    generation_error_lte_2_percent: (candidate.generation.error_rate ?? 1) <= 0.02,
    headroom_coverage_100_percent: candidate.retrieval.headroom_query_coverage_rate === 1,
    contract_pass_100_percent: candidate.generation.contract_pass_rate === 1,
    winner_delta_at_least_4: winnerDelta >= 4,
    candidate_worse_rate_within_3_points: (
      candidate.retrieval.candidate_worse_rate !== null
      && baseline.retrieval.candidate_worse_rate !== null
      && candidate.retrieval.candidate_worse_rate <= baseline.retrieval.candidate_worse_rate + 0.03
    ),
    mrr_not_lower: !!baselineMetrics && !!candidateMetrics
      && candidateMetrics.mrr_at_10 >= baselineMetrics.mrr_at_10,
    ndcg_not_lower: !!baselineMetrics && !!candidateMetrics
      && candidateMetrics.ndcg_at_10 >= baselineMetrics.ndcg_at_10,
  };
  // Semantic checks require a hand-authored audit that most runs do not have.
  // Scoring their absence as a failure made every unaudited run fail for a
  // reason unrelated to its quality, so they only participate once the audits
  // they read are actually present.
  const semanticChecksApply = candidate.semantic?.complete === true;
  if (semanticChecksApply) {
    checks.critical_winners_zero = candidate.semantic!.winner_counts.critical === 0;
    if (baseline.semantic?.complete === true) {
      checks.semantic_admitted_winners_increased =
        candidate.semantic!.semantic_admitted_winners
        >= baseline.semantic.semantic_admitted_winners;
    }
  }
  return {
    baseline_run: basename(baseline.run_dir),
    candidate_run: basename(candidate.run_dir),
    infrastructure_or_prompt_winner_delta: winnerDelta,
    headroom_conversion_delta: conversionDelta,
    semantic_checks_applied: semanticChecksApply,
    checks,
    passed: Object.values(checks).every(Boolean),
  };
}

const { values } = parseArgs({
  options: {
    "run-dir": { type: "string" },
    "semantic-audit": { type: "string" },
    "baseline-run-dir": { type: "string" },
    "baseline-semantic-audit": { type: "string" },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});
if (values.help) usage(0);
if (!values["run-dir"]) usage(1);

const runDir = resolve(values["run-dir"]);
const candidatesPath = join(runDir, "candidates.jsonl");
const candidatesBytes = readFileSync(candidatesPath);
const report = buildReport(
  runDir,
  parseDistillRecords(candidatesBytes.toString("utf8")),
  candidatesBytes,
  loadAudit(values["semantic-audit"]),
);
const output: Record<string, unknown> = { ...report };
if (values["baseline-run-dir"]) {
  const baselineDir = resolve(values["baseline-run-dir"]);
  const baselineBytes = readFileSync(join(baselineDir, "candidates.jsonl"));
  const baseline = buildReport(
    baselineDir,
    parseDistillRecords(baselineBytes.toString("utf8")),
    baselineBytes,
    loadAudit(values["baseline-semantic-audit"]),
  );
  if (JSON.stringify(baseline.retrieval.headroom_qids) !== JSON.stringify(report.retrieval.headroom_qids)) {
    throw new Error("Baseline and candidate runs do not contain the same ordered headroom qids");
  }
  output.comparison = compareRuns(baseline, report);
}
writeFileSync(join(runDir, "distill-report.json"), `${JSON.stringify(output, null, 2)}\n`, "utf8");
writeFileSync(
  join(runDir, "headroom-qids.txt"),
  `${report.retrieval.headroom_qids.join("\n")}\n`,
  "utf8",
);
console.log(JSON.stringify(output, null, 2));

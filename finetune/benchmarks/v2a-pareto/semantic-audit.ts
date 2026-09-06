#!/usr/bin/env node
/**
 * V2-A Pareto: Task 4 — Semantic Audit (spec section 9).
 *
 * Reads the selection-ledger.jsonl from Task 3, sends each winner to a
 * DeepSeek LLM judge, and records pass/fail/uncertain with reason codes.
 *
 * Environment variables:
 *   DISTILL_API_KEY       — DeepSeek API key (required)
 *   DISTILL_API_BASE_URL  — defaults to https://api.deepseek.com
 */

import { readFileSync, writeFileSync, appendFileSync, existsSync, renameSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const DEEPSEEK_BASE = "https://api.deepseek.com";

// --- Paths ---
const FINETUNE_ROOT = resolve(import.meta.dirname, "..", "..");
const V2A_ROOT = join(FINETUNE_ROOT, "data", "public-distill-v3-vh-pareto", "experiments", "public-main-v3-vh-pareto-prompt-v1");
const LEDGER_PATH = join(V2A_ROOT, "3-pareto-selection", "selection-ledger.jsonl");
const OUTPUT_DIR = join(V2A_ROOT, "4-semantic-audit");

// --- Judge prompt ---
const JUDGE_SYSTEM_PROMPT = `You are a semantic quality auditor for query expansion training data.
Your task is to check whether a generated expansion (hyde + vec lines) is faithful to the original query.

For each check, respond with one of: pass, fail, uncertain.

Checks:
1. entity_drift: Does the expansion change entities (names, brands, versions, numbers) from the query?
2. lost_constraint: Does the expansion drop a constraint from the query (e.g., "beginner", "Python only", "under $100")?
3. lost_negation: Does the expansion ignore or reverse a negation in the query?
4. unsupported_fact: Does the hyde hallucinate facts not implied by the query?
5. unsupported_causality: Does the hyde fabricate causal claims not in the query?
6. ambiguous_overinterpretation: Does the expansion resolve ambiguity in the query by guessing a specific interpretation?
7. unsupported_abbreviation_expansion: Does the expansion expand an abbreviation incorrectly?
8. non_complementary_vec: Are the vec lines essentially the same query (not complementary)?
9. intent_drift: Does the expansion change the intent of the original query?

Respond with a JSON object:
{
  "overall": "pass" | "fail" | "uncertain",
  "checks": {
    "entity_drift": "pass" | "fail" | "uncertain",
    ...
  },
  "failed_reasons": ["reason_code", ...]  // only for failed checks
}`;

const JUDGE_USER_TEMPLATE = `Original query: {{query}}

Generated expansion:
{{expansion}}

Evaluate the expansion against the original query using the 9 checks.`;

// --- Main ---

const { values } = parseArgs({
  options: {
    "model": { type: "string", default: "deepseek-chat" },
    "concurrency": { type: "string", default: "4" },
  },
});

const apiKey = process.env.DISTILL_API_KEY;
if (!apiKey) throw new Error("DISTILL_API_KEY environment variable is required");
const baseUrl = process.env.DISTILL_API_BASE_URL;
const model = values.model!;

// Load winners
const ledger = readFileSync(LEDGER_PATH, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
const winners = ledger.filter((r: any) => r.status === "winner" && r.winner);
console.error(`Loaded ${winners.length} winners from ${ledger.length} ledger entries`);

// Load canonical candidates for expansion text
const canonicalPath = join(V2A_ROOT, "1-canonicalization", "canonical-candidates.jsonl");
const canonicalById = new Map<string, any>();
for (const line of readFileSync(canonicalPath, "utf8").trim().split("\n").filter(Boolean)) {
  const c = JSON.parse(line);
  canonicalById.set(c.source_candidate_id, c);
}

// Build audit records
const auditRecords = winners.map((w: any) => {
  const candidate = canonicalById.get(w.winner.source_candidate_id);
  if (!candidate) throw new Error(`Candidate not found: ${w.winner.source_candidate_id}`);
  const expansion = candidate.canonical_output
    .map(([type, text]: [string, string]) => `${type}: ${text}`)
    .join("\n");
  return { ledger_entry: w, candidate, expansion };
});

console.error(`Prepared ${auditRecords.length} audit targets`);

// Output paths
const rawPath = join(OUTPUT_DIR, "semantic-audit.raw.jsonl");
const partialRawPath = `${rawPath}.partial`;
const resultPath = join(OUTPUT_DIR, "semantic-audit.json");
const manifestPath = join(OUTPUT_DIR, "judge-manifest.json");

if (existsSync(resultPath)) throw new Error("Audit result already exists. Remove it first.");

// Load partial progress
const doneIds = new Set<string>();
if (existsSync(partialRawPath)) {
  for (const line of readFileSync(partialRawPath, "utf8").trim().split("\n").filter(Boolean)) {
    const record = JSON.parse(line);
    doneIds.add(record.input_id);
  }
  console.error(`Resuming: ${doneIds.size} already audited`);
}

const pending = auditRecords.filter(r => !doneIds.has(r.ledger_entry.input_id));
console.error(`Pending: ${pending.length}`);

// Concurrency
const concurrency = Math.max(1, parseInt(values.concurrency!, 10) || 4);
function semaphore(limit: number) {
  let running = 0;
  const queue: (() => void)[] = [];
  const next = () => { if (queue.length > 0) { running++; queue.shift()!(); } };
  return {
    async run<T>(fn: () => Promise<T>): Promise<T> {
      if (running >= limit) await new Promise<void>(r => queue.push(r));
      running++;
      try { return await fn(); } finally { running--; next(); }
    },
  };
}
const pool = semaphore(concurrency);
const writeLock = semaphore(1);

async function judgeQuery(query: string, expansion: string): Promise<{ output: string; error?: string }> {
  const apiUrl = `${(baseUrl ?? DEEPSEEK_BASE).replace(/\/+$/, "")}/chat/completions`;
  const userContent = `Original query: ${query}\n\nGenerated expansion:\n${expansion}\n\nEvaluate the expansion against the original query using the 9 checks.`;

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: JUDGE_SYSTEM_PROMPT },
            { role: "user", content: userContent },
          ],
          max_tokens: 1024,
          response_format: { type: "json_object" },
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
      }

      const data = await response.json() as any;
      const content = data?.choices?.[0]?.message?.content;
      if (!content) throw new Error(`Empty response: ${JSON.stringify(data).slice(0, 200)}`);
      return { output: content };
    } catch (err) {
      if (attempt === 2) return { output: "", error: err instanceof Error ? err.message : String(err) };
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  return { output: "", error: "unreachable" };
}

let completed = 0;
const total = pending.length;

try {
  await Promise.all(pending.map(record => pool.run(async () => {
    const inputId = record.ledger_entry.input_id;
    const query = record.ledger_entry.winner
      ? record.ledger_entry.raw_metrics
      : record.ledger_entry.query;
    const sourceQuery = record.candidate.query;

    const result = await judgeQuery(sourceQuery, record.expansion);

    let parsed: any = null;
    let parseError: string | null = null;
    try {
      parsed = JSON.parse(result.output);
    } catch (e) {
      parseError = e instanceof Error ? e.message : String(e);
    }

    const auditRecord = {
      input_id: inputId,
      qid: record.ledger_entry.qid,
      source_id: record.ledger_entry.source_id,
      source_candidate_id: record.ledger_entry.winner.source_candidate_id,
      canonical_candidate_id: record.ledger_entry.winner.canonical_candidate_id,
      query: sourceQuery,
      expansion: record.expansion,
      raw_judge_output: result.output,
      parsed: parsed,
      parse_error: parseError,
      generation_error: result.error,
      overall: parsed?.overall ?? (parseError ? "parse_error" : "unknown"),
      failed_reasons: parsed?.failed_reasons ?? [],
      checks: parsed?.checks ?? {},
    };

    await writeLock.run(async () => {
      appendFileSync(partialRawPath, `${JSON.stringify(auditRecord)}\n`, "utf8");
    });

    completed++;
    if (completed % 10 === 0 || completed === total) {
      process.stderr.write(`Audit ${completed}/${total}\n`);
    }
  })));
} finally {
  // Finalize
  if (existsSync(partialRawPath)) renameSync(partialRawPath, rawPath);
}

// Summarize
const rawRecords = readFileSync(rawPath, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
const summary = {
  total: rawRecords.length,
  pass: rawRecords.filter((r: any) => r.overall === "pass").length,
  fail: rawRecords.filter((r: any) => r.overall === "fail").length,
  uncertain: rawRecords.filter((r: any) => r.overall === "uncertain").length,
  parse_error: rawRecords.filter((r: any) => r.overall === "parse_error").length,
  generation_error: rawRecords.filter((r: any) => r.generation_error).length,
  reason_codes: {} as Record<string, number>,
};

for (const r of rawRecords) {
  for (const reason of (r.failed_reasons || [])) {
    summary.reason_codes[reason] = (summary.reason_codes[reason] || 0) + 1;
  }
}

writeFileSync(resultPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
writeFileSync(manifestPath, `${JSON.stringify({
  schema_version: "qmd-v2a-semantic-audit-v1",
  judge_model: model,
  judge_prompt_version: "v2a-semantic-judge-v1",
  judge_prompt_sha256: createHash("sha256").update(JUDGE_SYSTEM_PROMPT + JUDGE_USER_TEMPLATE).digest("hex"),
  total_audited: summary.total,
  pass: summary.pass,
  fail: summary.fail,
  uncertain: summary.uncertain,
  parse_error: summary.parse_error,
  generation_error: summary.generation_error,
  reason_codes: summary.reason_codes,
}, null, 2)}\n`, "utf8");

console.error(`\nDone. ${summary.pass} pass, ${summary.fail} fail, ${summary.uncertain} uncertain`);
console.error(`Manifest: ${manifestPath}`);
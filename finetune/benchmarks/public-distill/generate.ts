#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import {
  OpenAiGenerationError,
  generateOpenAiChatExpansion,
  loadOpenAiPromptConfig,
  openAiChatCompletionsEndpoint,
} from "../lib/openai-teacher.js";

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function json(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

const { values } = parseArgs({
  options: {
    experiment: { type: "string", default: "public-smoke-v0" },
    root: { type: "string", default: "finetune/data/public-distill-v0" },
    pool: { type: "string", default: "smoke" },
    preflight: { type: "boolean", default: false },
  },
});
if (values.pool !== "smoke" && values.pool !== "main") throw new Error("--pool must be smoke or main");
const smokeOnly = values.pool === "smoke";
const expectedQueries = smokeOnly ? 30 : 2500;
const root = resolve(values.root);
const poolFilename = smokeOnly ? "pool-smoke.jsonl" : "pool-main.jsonl";
const poolPath = join(root, "prepared", poolFilename);
const poolBytes = readFileSync(poolPath);
const pool = poolBytes.toString("utf8").trim().split("\n").map(line => JSON.parse(line)) as {
  input_id: string;
  source_id: string;
  qid: string;
  query: string;
  sample_key: string;
}[];
if (pool.length !== expectedQueries) {
  throw new Error(`${values.pool} pool must contain ${expectedQueries} inputs, got ${pool.length}`);
}
let poolManifestSha256: string | null = null;
if (!smokeOnly) {
  const poolManifestPath = join(root, "prepared", "pool-main-manifest.json");
  const poolManifestBytes = readFileSync(poolManifestPath);
  const poolManifest = JSON.parse(poolManifestBytes.toString("utf8"));
  if (
    poolManifest.status !== "frozen_pre_generation"
    || poolManifest.total_count !== expectedQueries
    || poolManifest.pool_sha256 !== sha256(poolBytes)
    || poolManifest.smoke_overlap_count !== 0
    || poolManifest.smoke_normalized_query_overlap_count !== 0
    || poolManifest.holdout_overlap_count !== 0
    || poolManifest.unique_input_id_count !== expectedQueries
    || poolManifest.unique_normalized_query_count !== expectedQueries
  ) {
    throw new Error("Formal generation requires a valid frozen main pool manifest");
  }
  poolManifestSha256 = sha256(poolManifestBytes);
}
const provider = process.env.DISTILL_PROVIDER;
const model = process.env.DISTILL_MODEL;
const apiKey = process.env.DISTILL_API_KEY;
const baseUrl = process.env.DISTILL_API_BASE_URL;
const thinkingMode = process.env.DISTILL_THINKING_MODE;
if (provider !== "deepseek" || model !== "deepseek-v4-flash") {
  throw new Error("Public distillation requires DISTILL_PROVIDER=deepseek and DISTILL_MODEL=deepseek-v4-flash");
}
if (baseUrl !== "https://api.deepseek.com" || thinkingMode !== "disabled") {
  throw new Error("Public distillation requires the frozen DeepSeek base URL and disabled thinking mode");
}
if (!apiKey) throw new Error("DISTILL_API_KEY is required");
const prompt = loadOpenAiPromptConfig();
if (
  prompt.version !== "qmd-expansion-teacher-v2-deepseek-json4-semantic-safe"
  || basename(prompt.systemPromptFile ?? "") !== "deepseek-json4-v2-system.txt"
) {
  throw new Error("Public distillation requires the frozen DeepSeek JSON4 V2 Prompt");
}
const maxOutputTokens = Number(process.env.DISTILL_MAX_OUTPUT_TOKENS ?? "4096");
if (maxOutputTokens !== 4096) throw new Error("Public distillation requires DISTILL_MAX_OUTPUT_TOKENS=4096");

const dirtyEntries = execFileSync("git", ["status", "--porcelain", "--untracked-files=all"], { encoding: "utf8" })
  .trim().split("\n").filter(Boolean)
  .filter(line => !line.slice(3).startsWith("finetune/data/public-distill-v0/"));
if (!smokeOnly && dirtyEntries.length > 0) {
  throw new Error(`Formal generation requires versioned, clean source files; dirty entries:\n${dirtyEntries.join("\n")}`);
}

const runDir = join(root, "experiments", values.experiment);
const candidatesPath = join(runDir, "candidates.jsonl");
const partialPath = `${candidatesPath}.partial`;
const manifestPath = join(runDir, "run-manifest.json");
if (values.preflight) {
  process.stdout.write(`${JSON.stringify({
    status: "preflight_ok",
    experiment_id: values.experiment,
    pool: values.pool,
    pool_sha256: sha256(poolBytes),
    pool_manifest_sha256: poolManifestSha256,
    input_queries: pool.length,
    candidates: pool.length * 4,
    teacher_provider: provider,
    teacher_model: model,
    thinking_mode: thinkingMode,
    prompt_version: prompt.version,
    prompt_sha256: prompt.sha256,
    qmd_dirty: dirtyEntries.length > 0,
  }, null, 2)}\n`);
  process.exit(0);
}
mkdirSync(runDir, { recursive: true });
if (existsSync(candidatesPath)) throw new Error(`Candidate artifact already exists: ${candidatesPath}`);
const completed = existsSync(partialPath)
  ? readFileSync(partialPath, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line))
  : [];
for (let index = 0; index < completed.length; index++) {
  if (completed[index].input_id !== pool[index]?.input_id) throw new Error("Partial artifact is not a pool prefix");
}
if (!existsSync(manifestPath)) {
  json(manifestPath, {
    schema_version: "qmd-public-distill-run-v0",
    experiment_id: values.experiment,
    smoke_only: smokeOnly,
    final_sft_eligible: false,
    created_at: new Date().toISOString(),
    qmd_commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    qmd_dirty: dirtyEntries.length > 0,
    pool_path: `prepared/${poolFilename}`,
    pool_sha256: sha256(poolBytes),
    pool_manifest_path: smokeOnly ? null : "prepared/pool-main-manifest.json",
    pool_manifest_sha256: poolManifestSha256,
    teacher_provider: provider,
    teacher_model: model,
    api_base_url: baseUrl,
    api_endpoint: openAiChatCompletionsEndpoint(baseUrl),
    thinking_mode: thinkingMode,
    temperature: "provider_default",
    top_p: "provider_default",
    max_output_tokens: maxOutputTokens,
    prompt_version: prompt.version,
    prompt_sha256: prompt.sha256,
    system_prompt_file: prompt.systemPromptFile,
    user_prompt_template_file: prompt.userPromptTemplateFile,
    candidate_count_per_query: 4,
    max_attempts_per_candidate: 3,
  });
}

for (const input of pool.slice(completed.length)) {
  const candidates = [];
  for (let candidateIndex = 0; candidateIndex < 4; candidateIndex++) {
    const startedAt = performance.now();
    try {
      const result = await generateOpenAiChatExpansion({
        apiKey,
        baseUrl,
        model,
        query: input.query,
        maxOutputTokens,
        prompt,
        thinkingMode: "disabled",
        maxAttempts: 3,
      });
      candidates.push({
        candidate_index: candidateIndex,
        generation_status: "ok",
        raw_output: result.raw_output,
        parsed_output: result.parsed_output,
        generation_error: null,
        generation_error_type: null,
        generation_diagnostics: null,
        latency_ms: performance.now() - startedAt,
        contract: null,
        repeat_check: null,
        retrieval: null,
      });
    } catch (error) {
      const known = error instanceof OpenAiGenerationError ? error : null;
      candidates.push({
        candidate_index: candidateIndex,
        generation_status: "generation_error",
        raw_output: "",
        parsed_output: [],
        generation_error: error instanceof Error ? error.message : String(error),
        generation_error_type: known?.code ?? null,
        generation_diagnostics: known?.diagnostics ?? null,
        latency_ms: performance.now() - startedAt,
        contract: null,
        repeat_check: null,
        retrieval: null,
      });
    }
    process.stderr.write(`Generated ${input.input_id} candidate ${candidateIndex + 1}/4\n`);
  }
  const record = {
    schema_version: "qmd-public-distill-v0",
    ...input,
    smoke_only: smokeOnly,
    raw: { status: "pending", metrics: null, top_30_doc_ids: [] },
    candidates,
    selected_candidate_index: null,
    selection_status: "pending",
  };
  appendFileSync(partialPath, `${JSON.stringify(record)}\n`, "utf8");
}
renameSync(partialPath, candidatesPath);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
manifest.candidates_sha256 = sha256(readFileSync(candidatesPath));
manifest.generated_queries = pool.length;
manifest.generated_candidates = pool.length * 4;
manifest.generation_errors = readFileSync(candidatesPath, "utf8").split("\n").filter(Boolean)
  .map(line => JSON.parse(line)).flatMap(record => record.candidates)
  .filter(candidate => candidate.generation_status !== "ok").length;
json(manifestPath, manifest);
process.stdout.write(`${JSON.stringify({ run_dir: runDir, generated_queries: pool.length }, null, 2)}\n`);

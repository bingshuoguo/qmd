#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parseGeneratedExpansion } from "../../../src/bench/expansions.js";
import { validateBenchmarkRunName } from "../../../src/bench/run-name.js";
import { applyLlamaEnvMitigation, writeJson } from "../lib/cli.js";
import {
  SCIFACT_DISTILL_CANDIDATE_COUNT,
  assertQidPrefix,
  parseDistillRecords,
  parseSciFactDistillSplit,
  parseSciFactSourceQueries,
  sha256,
  type DistillCandidate,
  type SciFactDistillRecord,
} from "../lib/distill.js";
import {
  DEEPSEEK_API_BASE_URL,
  OPENAI_DEFAULT_MAX_OUTPUT_TOKENS,
  OPENAI_LEGACY_MAX_OUTPUT_TOKENS,
  OPENAI_REQUEST_TIMEOUT_MS,
  OPENAI_REASONING_EFFORTS,
  OPENAI_RESPONSES_ENDPOINT,
  OpenAiGenerationError,
  generateOpenAiChatExpansion,
  generateOpenAiExpansion,
  loadOpenAiPromptConfig,
  openAiChatCompletionsEndpoint,
  type OpenAiReasoningEffort,
} from "../lib/openai-teacher.js";

type TeacherProvider = "llama.cpp" | "openai" | "openai-compatible" | "deepseek";

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run distill:generate -- --experiment <id> --model <id-or-path> \\",
    "    [--provider llama.cpp|openai|openai-compatible|deepseek] \\",
    "    [--thinking-mode enabled|disabled] \\",
    "    [--reasoning-effort none|low|medium|high|xhigh|max] \\",
    "    [--max-output-tokens 4096] \\",
    "    [--root finetune/data/distillation/scifact-v1] \\",
    "    [--queries archive/scifact/queries.jsonl] [--max-queries N|--qids-file PATH] \\",
    "    [--seed-run <pilot-run-dir>] [--retry-generation-errors] [--force]",
    "",
    "The model can be supplied by --model or DISTILL_MODEL.",
    "Credentials are read from DISTILL_API_KEY, falling back to OPENAI_API_KEY.",
    "openai-compatible additionally requires DISTILL_RESPONSES_ENDPOINT.",
    "deepseek uses DISTILL_API_BASE_URL (default https://api.deepseek.com).",
    "Prompt configuration uses DISTILL_PROMPT_VERSION plus either inline",
    "DISTILL_SYSTEM_PROMPT/DISTILL_USER_PROMPT_TEMPLATE or their *_FILE variants.",
    "DISTILL_MAX_OUTPUT_TOKENS is used when --max-output-tokens is omitted.",
  ].join("\n");
  (exitCode === 0 ? console.log : console.error)(message);
  process.exit(exitCode);
}

function git(args: string[]): string {
  return execFileSync("git", args, { encoding: "utf8" }).trim();
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function renderProgress(options: {
  completed: number;
  total: number;
  resumed: number;
  startedAt: number;
  qid: string | null;
  candidateIndex: number | null;
  lastCandidateMs: number | null;
  generationErrors: number;
}): void {
  if (!process.stderr.isTTY) return;
  const width = 24;
  const ratio = options.total === 0 ? 1 : Math.min(1, options.completed / options.total);
  const filled = Math.round(ratio * width);
  const elapsedMs = performance.now() - options.startedAt;
  const generatedThisRun = options.completed - options.resumed;
  const remaining = options.total - options.completed;
  const eta = generatedThisRun === 0
    ? "--"
    : formatDuration((elapsedMs / generatedThisRun) * remaining);
  const last = options.lastCandidateMs === null ? "--" : formatDuration(options.lastCandidateMs);
  const current = options.qid === null || options.candidateIndex === null
    ? ""
    : ` qid=${options.qid} candidate=${options.candidateIndex + 1}/${SCIFACT_DISTILL_CANDIDATE_COUNT}`;
  const errors = options.generationErrors === 0 ? "" : ` errors=${options.generationErrors}`;
  process.stderr.write(
    `\x1b[2K\rDistill [${"=".repeat(filled)}${"-".repeat(width - filled)}]`
    + ` ${Math.round(ratio * 100).toString().padStart(3)}%`
    + ` ${options.completed}/${options.total} candidates${current}`
    + ` elapsed=${formatDuration(elapsedMs)} eta=${eta} last=${last}${errors}`
    + (options.completed === options.total ? "\n" : ""),
  );
}

function writeRecords(path: string, records: readonly SciFactDistillRecord[]): void {
  const temporaryPath = `${path}.tmp`;
  const text = records.map(record => JSON.stringify(record)).join("\n") + "\n";
  writeFileSync(temporaryPath, text, "utf8");
  renameSync(temporaryPath, path);
}

function parsePositiveInteger(value: string | undefined, label: string): number | null {
  if (value === undefined) return null;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function generationFailure(error: unknown): Pick<
  DistillCandidate,
  | "generation_status"
  | "raw_output"
  | "parsed_output"
  | "generation_error"
  | "generation_error_type"
  | "generation_diagnostics"
> {
  if (error instanceof OpenAiGenerationError) {
    return {
      generation_status: "generation_error",
      raw_output: "",
      parsed_output: [],
      generation_error: error.message,
      generation_error_type: error.code,
      generation_diagnostics: error.diagnostics,
    };
  }
  return {
    generation_status: "generation_error",
    raw_output: "",
    parsed_output: [],
    generation_error: error instanceof Error ? error.message : String(error),
    generation_error_type: null,
    generation_diagnostics: null,
  };
}

const { values } = parseArgs({
  options: {
    experiment: { type: "string" },
    model: { type: "string", default: process.env.DISTILL_MODEL },
    provider: { type: "string", default: process.env.DISTILL_PROVIDER ?? "llama.cpp" },
    "reasoning-effort": { type: "string", default: "low" },
    "thinking-mode": { type: "string", default: process.env.DISTILL_THINKING_MODE ?? "disabled" },
    "max-output-tokens": { type: "string" },
    root: { type: "string", default: "finetune/data/distillation/scifact-v1" },
    queries: { type: "string", default: "archive/scifact/queries.jsonl" },
    "max-queries": { type: "string" },
    "qids-file": { type: "string" },
    "seed-run": { type: "string" },
    "retry-generation-errors": { type: "boolean", default: false },
    force: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});
if (values.help) usage(0);
if (!values.experiment || !values.model) usage(1);
const retryGenerationErrors = values["retry-generation-errors"];
if (retryGenerationErrors && values.force) {
  throw new Error("--retry-generation-errors cannot be combined with --force");
}
if (
  values.provider !== "llama.cpp"
  && values.provider !== "openai"
  && values.provider !== "openai-compatible"
  && values.provider !== "deepseek"
) {
  throw new Error(`Unsupported teacher provider: ${values.provider}`);
}
const teacherProvider = values.provider as TeacherProvider;
const usesRemoteApi = teacherProvider !== "llama.cpp";
const usesResponsesApi = teacherProvider === "openai" || teacherProvider === "openai-compatible";
const usesChatCompletionsApi = teacherProvider === "deepseek";
if (!OPENAI_REASONING_EFFORTS.includes(values["reasoning-effort"] as OpenAiReasoningEffort)) {
  throw new Error(`Unsupported OpenAI reasoning effort: ${values["reasoning-effort"]}`);
}
const reasoningEffort = values["reasoning-effort"] as OpenAiReasoningEffort;
if (values["thinking-mode"] !== "enabled" && values["thinking-mode"] !== "disabled") {
  throw new Error("--thinking-mode must be enabled or disabled");
}
const thinkingMode = usesChatCompletionsApi
  ? values["thinking-mode"] as "enabled" | "disabled"
  : null;
if (
  thinkingMode === "enabled"
  && reasoningEffort !== "high"
  && reasoningEffort !== "max"
) {
  throw new Error("DeepSeek thinking mode supports --reasoning-effort high or max");
}
const manifestReasoningEffort = usesResponsesApi
  ? reasoningEffort
  : thinkingMode === "enabled" ? reasoningEffort : null;
const prompt = usesRemoteApi ? loadOpenAiPromptConfig() : null;
const configuredMaxOutputTokens = parsePositiveInteger(
  values["max-output-tokens"] ?? process.env.DISTILL_MAX_OUTPUT_TOKENS,
  values["max-output-tokens"] === undefined
    ? "DISTILL_MAX_OUTPUT_TOKENS"
    : "--max-output-tokens",
);
const openAiApiKey = usesRemoteApi
  ? process.env.DISTILL_API_KEY ?? process.env.OPENAI_API_KEY
  : undefined;
if (usesRemoteApi && !openAiApiKey) {
  throw new Error("DISTILL_API_KEY or OPENAI_API_KEY is required for a remote API provider");
}
const responsesEndpoint = teacherProvider === "openai-compatible"
  ? process.env.DISTILL_RESPONSES_ENDPOINT
  : OPENAI_RESPONSES_ENDPOINT;
if (teacherProvider === "openai-compatible" && !responsesEndpoint) {
  throw new Error("DISTILL_RESPONSES_ENDPOINT is required for --provider openai-compatible");
}
const apiBaseUrl = usesChatCompletionsApi
  ? process.env.DISTILL_API_BASE_URL ?? DEEPSEEK_API_BASE_URL
  : null;
const apiEndpoint = usesChatCompletionsApi
  ? openAiChatCompletionsEndpoint(apiBaseUrl!)
  : usesResponsesApi ? responsesEndpoint! : null;
const experimentId = validateBenchmarkRunName(values.experiment);
if (values["max-queries"] !== undefined && values["qids-file"] !== undefined) {
  throw new Error("--max-queries cannot be combined with --qids-file");
}
const maxQueries = values["max-queries"] === undefined
  ? null
  : Number.parseInt(values["max-queries"], 10);
if (maxQueries !== null && (!Number.isSafeInteger(maxQueries) || maxQueries <= 0)) {
  throw new Error("--max-queries must be a positive integer");
}
const root = resolve(values.root);
const splitPath = join(root, "split.json");
const queriesPath = resolve(values.queries);
const qidsPath = values["qids-file"] ? resolve(values["qids-file"]) : null;
const runDir = join(root, "runs", experimentId);
const manifestPath = join(runDir, "manifest.json");
const outputPath = join(runDir, "candidates.jsonl");
const partialPath = `${outputPath}.partial`;
const scoredPartialPath = `${outputPath}.scored.partial`;
const seedRunDir = values["seed-run"] ? resolve(values["seed-run"]) : null;
const splitBytes = readFileSync(splitPath);
const queriesBytes = readFileSync(queriesPath);
const split = parseSciFactDistillSplit(splitBytes.toString("utf8"));
const availableQueryCount = split.train_qids.length + split.val_qids.length;
if (maxQueries !== null && maxQueries > availableQueryCount) {
  throw new Error(`--max-queries=${maxQueries} exceeds ${availableQueryCount} available queries`);
}
if (sha256(queriesBytes) !== split.source.queries_sha256) {
  throw new Error("queries.jsonl does not match split.json");
}
const queryById = new Map(
  parseSciFactSourceQueries(queriesBytes.toString("utf8")).map(query => [query.qid, query.query]),
);
const allOrdered = [
  ...split.train_qids.map(qid => ({ qid, split: "train" as const })),
  ...split.val_qids.map(qid => ({ qid, split: "val" as const })),
];
const itemByQid = new Map(allOrdered.map(item => [item.qid, item]));
let qidsSha256: string | null = null;
let ordered = maxQueries === null ? allOrdered : allOrdered.slice(0, maxQueries);
if (qidsPath) {
  const qidsBytes = readFileSync(qidsPath);
  qidsSha256 = sha256(qidsBytes);
  const qids = qidsBytes.toString("utf8").split(/\r?\n/).map(qid => qid.trim()).filter(Boolean);
  if (qids.length === 0) throw new Error("--qids-file must contain at least one qid");
  if (new Set(qids).size !== qids.length) throw new Error("--qids-file contains duplicate qids");
  ordered = qids.map(qid => {
    const item = itemByQid.get(qid);
    if (!item) throw new Error(`--qids-file references qid outside the frozen split: ${qid}`);
    return item;
  });
}
for (const { qid } of ordered) {
  if (!queryById.has(qid)) throw new Error(`split.json references missing query "${qid}"`);
}
if (existsSync(outputPath) && !values.force && !retryGenerationErrors) {
  throw new Error(`Candidate artifact already exists: ${outputPath}`);
}
if (
  retryGenerationErrors
  && !seedRunDir
  && !existsSync(outputPath)
  && !existsSync(partialPath)
) {
  throw new Error(`Candidate artifact does not exist: ${outputPath}`);
}

mkdirSync(runDir, { recursive: true });
if (values.force) {
  rmSync(outputPath, { force: true });
  rmSync(partialPath, { force: true });
  rmSync(scoredPartialPath, { force: true });
  rmSync(manifestPath, { force: true });
}
const splitSha256 = sha256(splitBytes);
let manifest: Record<string, unknown>;
let maxOutputTokens: number | null = usesRemoteApi
  ? configuredMaxOutputTokens ?? OPENAI_DEFAULT_MAX_OUTPUT_TOKENS
  : null;
if (existsSync(manifestPath)) {
  manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
  const manifestProvider = manifest.teacher_provider ?? "llama.cpp";
  const manifestMaxOutputTokens = usesRemoteApi
    ? manifest.max_output_tokens ?? OPENAI_LEGACY_MAX_OUTPUT_TOKENS
    : null;
  if (usesRemoteApi && configuredMaxOutputTokens === null) {
    maxOutputTokens = manifestMaxOutputTokens as number;
  }
  if (
    manifest.experiment_id !== experimentId
    || manifest.teacher_model !== values.model
    || manifest.split_sha256 !== splitSha256
    || manifestProvider !== teacherProvider
    || (manifest.query_limit ?? null) !== maxQueries
    || (manifest.qids_sha256 ?? null) !== qidsSha256
    || (
      usesRemoteApi
      && (
        manifest.reasoning_effort !== manifestReasoningEffort
        || (manifest.thinking_mode ?? null) !== thinkingMode
        || manifest.prompt_version !== prompt!.version
        || manifest.api_endpoint !== apiEndpoint
        || manifestMaxOutputTokens !== maxOutputTokens
        || (
          manifest.prompt_sha256 !== undefined
          && manifest.prompt_sha256 !== prompt!.sha256
        )
      )
    )
  ) {
    throw new Error("Existing run manifest does not match requested generation run");
  }
} else {
  manifest = {
    version: "scifact-distill-run-v1",
    experiment_id: experimentId,
    created_at: new Date().toISOString(),
    qmd_commit: git(["rev-parse", "HEAD"]),
    qmd_dirty: git(["status", "--porcelain"]).length > 0,
    split_path: basename(splitPath),
    split_sha256: splitSha256,
    teacher_provider: teacherProvider,
    teacher_model: values.model,
    generation_method: usesResponsesApi
      ? "OpenAI Responses API Structured Outputs"
      : usesChatCompletionsApi
        ? "OpenAI-compatible Chat Completions JSON Output"
        : "LlamaCpp.generateQueryExpansionRaw",
    prompt_version: usesRemoteApi ? prompt!.version : null,
    prompt_sha256: usesRemoteApi ? prompt!.sha256 : null,
    prompt_source: usesRemoteApi ? prompt!.source : null,
    system_prompt_file: usesRemoteApi ? prompt!.systemPromptFile : null,
    user_prompt_template_file: usesRemoteApi ? prompt!.userPromptTemplateFile : null,
    system_prompt: usesRemoteApi ? prompt!.systemPrompt : null,
    user_prompt_template: usesRemoteApi ? prompt!.userPromptTemplate : null,
    api_base_url: apiBaseUrl,
    api_endpoint: apiEndpoint,
    reasoning_effort: manifestReasoningEffort,
    thinking_mode: thinkingMode,
    max_output_tokens: maxOutputTokens,
    request_timeout_ms: usesRemoteApi ? OPENAI_REQUEST_TIMEOUT_MS : null,
    max_attempts_per_candidate: usesRemoteApi ? 3 : 1,
    candidates_per_query: SCIFACT_DISTILL_CANDIDATE_COUNT,
    query_limit: maxQueries,
    qids_file: qidsPath ? basename(qidsPath) : null,
    qids_sha256: qidsSha256,
    qids: qidsPath ? ordered.map(item => item.qid) : null,
    generation_status: "in_progress",
  };
}
manifest.max_attempts_per_candidate = usesRemoteApi ? 3 : 1;
if (usesRemoteApi) {
  manifest.max_output_tokens = maxOutputTokens;
  manifest.prompt_version = prompt!.version;
  manifest.prompt_sha256 = prompt!.sha256;
  manifest.prompt_source = prompt!.source;
  manifest.system_prompt_file = prompt!.systemPromptFile;
  manifest.user_prompt_template_file = prompt!.userPromptTemplateFile;
  manifest.system_prompt = prompt!.systemPrompt;
  manifest.user_prompt_template = prompt!.userPromptTemplate;
  manifest.api_base_url = apiBaseUrl;
  manifest.api_endpoint = apiEndpoint;
  manifest.reasoning_effort = manifestReasoningEffort;
  manifest.thinking_mode = thinkingMode;
}
writeJson(manifestPath, manifest);

if (!existsSync(partialPath) && seedRunDir) {
  const seedManifestPath = join(seedRunDir, "manifest.json");
  const seedCandidatesPath = join(seedRunDir, "candidates.jsonl");
  const seedManifest = JSON.parse(readFileSync(seedManifestPath, "utf8")) as Record<string, unknown>;
  if (
    seedManifest.split_sha256 !== splitSha256
    || seedManifest.teacher_provider !== teacherProvider
    || seedManifest.teacher_model !== values.model
    || seedManifest.prompt_version !== manifest.prompt_version
    || seedManifest.api_endpoint !== manifest.api_endpoint
    || seedManifest.reasoning_effort !== manifest.reasoning_effort
    || (seedManifest.thinking_mode ?? null) !== (manifest.thinking_mode ?? null)
    || (
      seedManifest.prompt_sha256 !== undefined
      && seedManifest.prompt_sha256 !== manifest.prompt_sha256
    )
  ) {
    throw new Error("Seed run does not match this generation configuration");
  }
  const seeded = parseDistillRecords(readFileSync(seedCandidatesPath, "utf8")).map(record => ({
    qid: record.qid,
    split: record.split,
    query: record.query,
    raw_metrics: null,
    candidates: record.candidates.map(candidate => ({
      candidate_index: candidate.candidate_index,
      generation_status: candidate.generation_status,
      raw_output: candidate.raw_output,
      parsed_output: candidate.parsed_output,
      generation_error: candidate.generation_error,
      generation_error_type: candidate.generation_error_type ?? null,
      generation_diagnostics: candidate.generation_diagnostics ?? null,
      contract: null,
      metrics: null,
    })),
    selected_candidate_index: null,
    selection_status: "pending" as const,
  }));
  assertQidPrefix("Seed candidate artifact", seeded, ordered);
  if (seeded.some((record, index) => (
    record.split !== ordered[index]!.split || record.query !== queryById.get(record.qid)
  ))) {
    throw new Error("Seed candidate artifact query or split does not match the frozen source");
  }
  writeFileSync(partialPath, seeded.map(record => JSON.stringify(record)).join("\n") + "\n", "utf8");
  manifest.seed_experiment_id = seedManifest.experiment_id;
  manifest.seed_candidates_sha256 = sha256(readFileSync(seedCandidatesPath));
  manifest.seeded_queries = seeded.length;
  manifest.seed_max_output_tokens = seedManifest.max_output_tokens
    ?? (usesRemoteApi ? OPENAI_LEGACY_MAX_OUTPUT_TOKENS : null);
  manifest.retry_seed_generation_errors = retryGenerationErrors;
  writeJson(manifestPath, manifest);
}
if (retryGenerationErrors && !existsSync(partialPath)) {
  writeFileSync(partialPath, readFileSync(outputPath));
}
if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");
const partialText = readFileSync(partialPath, "utf8");
const records = partialText.trim() ? parseDistillRecords(partialText) : [];
assertQidPrefix("Partial candidate artifact", records, ordered);
if (retryGenerationErrors && records.length !== ordered.length) {
  throw new Error("Retry candidate artifact does not contain every requested query");
}
const retryCandidateCount = records.reduce(
  (count, record) => count + record.candidates.filter(
    candidate => candidate.generation_status === "generation_error",
  ).length,
  0,
);
const totalCandidates = retryGenerationErrors
  ? retryCandidateCount
  : ordered.length * SCIFACT_DISTILL_CANDIDATE_COUNT;
const resumedCandidates = retryGenerationErrors
  ? 0
  : records.length * SCIFACT_DISTILL_CANDIDATE_COUNT;
const progressStartedAt = performance.now();
let progressLineOpen = false;
let generationErrors = retryGenerationErrors
  ? 0
  : retryCandidateCount;
const reportProgress = (
  completed: number,
  qid: string | null,
  candidateIndex: number | null,
  lastCandidateMs: number | null,
): void => {
  try {
    renderProgress({
      completed,
      total: totalCandidates,
      resumed: resumedCandidates,
      startedAt: progressStartedAt,
      qid,
      candidateIndex,
      lastCandidateMs,
      generationErrors,
    });
    progressLineOpen = !!process.stderr.isTTY && completed < totalCandidates;
  } catch {
    // Progress reporting must never fail or corrupt a generation run.
  }
};
reportProgress(resumedCandidates, null, null, null);

let disposeTeacher = async (): Promise<void> => {};
let generateCandidate: (query: string) => Promise<Pick<
  DistillCandidate,
  | "generation_status"
  | "raw_output"
  | "parsed_output"
  | "generation_error"
  | "generation_error_type"
  | "generation_diagnostics"
>>;
if (usesRemoteApi) {
  generateCandidate = async query => {
    const generated = usesResponsesApi
      ? await generateOpenAiExpansion({
        apiKey: openAiApiKey!,
        endpoint: responsesEndpoint,
        model: values.model!,
        query,
        reasoningEffort,
        maxOutputTokens: maxOutputTokens!,
        prompt: prompt!,
        maxAttempts: 3,
      })
      : await generateOpenAiChatExpansion({
        apiKey: openAiApiKey!,
        baseUrl: apiBaseUrl!,
        model: values.model!,
        query,
        maxOutputTokens: maxOutputTokens!,
        prompt: prompt!,
        thinkingMode: thinkingMode!,
        reasoningEffort: manifestReasoningEffort as "high" | "max" | undefined,
        maxAttempts: 3,
      });
    return {
      generation_status: "ok",
      raw_output: generated.raw_output,
      parsed_output: generated.parsed_output,
      generation_error: null,
      generation_error_type: null,
      generation_diagnostics: null,
    };
  };
} else {
  applyLlamaEnvMitigation();
  const { LlamaCpp } = await import("../../../src/llm.js");
  const llm = new LlamaCpp({ generateModel: values.model, inactivityTimeoutMs: 0 });
  disposeTeacher = () => llm.dispose();
  generateCandidate = async query => {
    const parsed = parseGeneratedExpansion(query, await llm.generateQueryExpansionRaw(query));
    return {
      generation_status: parsed.status,
      raw_output: parsed.raw_output,
      parsed_output: parsed.output,
      generation_error: parsed.error,
      generation_error_type: null,
      generation_diagnostics: null,
    };
  };
}
if (retryGenerationErrors) {
  let retriedCandidates = 0;
  for (const record of records) {
    record.raw_metrics = null;
    record.selected_candidate_index = null;
    record.selection_status = "pending";
    for (const candidate of record.candidates) candidate.metrics = null;
  }
  rmSync(scoredPartialPath, { force: true });
  try {
    for (const record of records) {
      const failedCandidates = record.candidates.filter(
        candidate => candidate.generation_status === "generation_error",
      );
      if (failedCandidates.length === 0) continue;
      for (const candidate of failedCandidates) {
        const candidateStartedAt = performance.now();
        try {
          const generated = await generateCandidate(record.query);
          Object.assign(candidate, {
            ...generated,
            contract: null,
            metrics: null,
          });
        } catch (error) {
          generationErrors++;
          Object.assign(candidate, {
            ...generationFailure(error),
            contract: null,
            metrics: null,
          });
        }
        retriedCandidates++;
        reportProgress(
          retriedCandidates,
          record.qid,
          candidate.candidate_index,
          performance.now() - candidateStartedAt,
        );
      }
      writeRecords(partialPath, records);
      if (!process.stderr.isTTY) {
        console.error(
          `Retried ${retriedCandidates}/${retryCandidateCount} candidates qid=${record.qid}`,
        );
      }
    }
  } finally {
    if (progressLineOpen) process.stderr.write("\n");
    await disposeTeacher();
  }

  const remainingGenerationErrors = records.reduce(
    (count, record) => count + record.candidates.filter(
      candidate => candidate.generation_status === "generation_error",
    ).length,
    0,
  );
  renameSync(partialPath, outputPath);
  manifest.generation_status = "completed";
  manifest.candidates_sha256 = sha256(readFileSync(outputPath));
  manifest.validated_candidates_sha256 = null;
  manifest.scored_candidates_sha256 = null;
  manifest.selection_counts = null;
  manifest.pilot_gate = null;
  manifest.retry_generation_errors = {
    completed_at: new Date().toISOString(),
    attempted_candidates: retryCandidateCount,
    recovered_candidates: retryCandidateCount - remainingGenerationErrors,
    remaining_generation_errors: remainingGenerationErrors,
  };
  writeJson(manifestPath, manifest);
  console.log(JSON.stringify({
    run_dir: runDir,
    attempted_candidates: retryCandidateCount,
    recovered_candidates: retryCandidateCount - remainingGenerationErrors,
    remaining_generation_errors: remainingGenerationErrors,
  }, null, 2));
} else {
  try {
  for (const [offset, item] of ordered.slice(records.length).entries()) {
    const query = queryById.get(item.qid)!;
    const candidates: DistillCandidate[] = [];
    for (let candidateIndex = 0; candidateIndex < SCIFACT_DISTILL_CANDIDATE_COUNT; candidateIndex++) {
      const candidateStartedAt = performance.now();
      try {
        const generated = await generateCandidate(query);
        candidates.push({
          candidate_index: candidateIndex,
          ...generated,
          contract: null,
          metrics: null,
        });
      } catch (error) {
        generationErrors++;
        candidates.push({
          candidate_index: candidateIndex,
          ...generationFailure(error),
          contract: null,
          metrics: null,
        });
      }
      reportProgress(
        records.length * SCIFACT_DISTILL_CANDIDATE_COUNT + candidates.length,
        item.qid,
        candidateIndex,
        performance.now() - candidateStartedAt,
      );
    }
    const record: SciFactDistillRecord = {
      qid: item.qid,
      split: item.split,
      query,
      raw_metrics: null,
      candidates,
      selected_candidate_index: null,
      selection_status: "pending",
    };
    records.push(record);
    appendFileSync(partialPath, `${JSON.stringify(record)}\n`, "utf8");
    const completed = records.length;
    if (!process.stderr.isTTY) {
      console.error(
        `Generated ${completed}/${ordered.length} qid=${item.qid} (${offset + 1} new this run)`,
      );
    }
  }
  } finally {
    if (progressLineOpen) process.stderr.write("\n");
    await disposeTeacher();
  }

  const completedText = readFileSync(partialPath, "utf8");
  const completedRecords = parseDistillRecords(completedText);
  if (completedRecords.length !== ordered.length) {
    throw new Error("Candidate generation did not complete");
  }
  renameSync(partialPath, outputPath);
  manifest.generation_status = "completed";
  manifest.candidates_sha256 = sha256(readFileSync(outputPath));
  manifest.generated_queries = completedRecords.length;
  manifest.total_available_queries = allOrdered.length;
  manifest.query_limit = maxQueries;
  writeJson(manifestPath, manifest);
  console.log(JSON.stringify({
    run_dir: runDir,
    generated_queries: completedRecords.length,
  }, null, 2));
}

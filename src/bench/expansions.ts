import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import {
  parseExpansionsJsonlVersioned,
  parseQueriesJsonl,
} from "./qrels.js";
import type {
  BenchmarkExpansion,
  BenchmarkExpansionRecord,
  BenchmarkExpansionRecordVh,
  ExpansionGenerationMeta,
  ExpansionSchemaVersion,
  ExpansionStatus,
} from "./types.js";
import { validateBenchmarkRunName } from "./run-name.js";
import {
  parseGeneratedExpansionLines,
  parseVhExpansionResponse,
} from "../query-expansion-parser.js";

type ParsedGeneratedExpansion = Pick<
  BenchmarkExpansionRecord,
  "status" | "raw_output" | "output" | "fallback_used" | "error"
>;

function formatError(rawOutput: string, error: string): ParsedGeneratedExpansion {
  return {
    status: "format_error",
    raw_output: rawOutput,
    output: [],
    fallback_used: false,
    error,
  };
}

export function parseGeneratedExpansion(
  _query: string,
  rawOutput: string,
): ParsedGeneratedExpansion {
  const parsed = parseGeneratedExpansionLines(rawOutput);
  if (parsed.diagnostics[0]?.code === "RUNTIME_EMPTY_OUTPUT") {
    return formatError(rawOutput, "model returned empty output");
  }
  if (parsed.diagnostics.length > 0) {
    const line = parsed.diagnostics[0]?.line ?? 1;
    return formatError(rawOutput, `line ${line} is not a typed expansion`);
  }

  return {
    status: "ok",
    raw_output: rawOutput,
    output: parsed.output as BenchmarkExpansion[],
    fallback_used: false,
    error: null,
  };
}

/**
 * Strict V2-VH parse with the serving fallback policy: any format failure
 * records the real error but sets `fallback_used=true` with empty output, so
 * the shared retrieval path executes the raw query.
 *
 * A complete-looking response only counts as ok when the engine reported a
 * natural stop ("eogToken"): a `maxTokens`/abort response whose tail happens
 * to parse is recorded as a failure, never silently accepted.
 */
export function parseVhExpansion(
  query: string,
  qid: string,
  rawOutput: string,
  meta: ExpansionGenerationMeta,
): BenchmarkExpansionRecordVh {
  const parsed = parseVhExpansionResponse(rawOutput);
  // VH fallback policy: every format failure (including truncation) falls
  // back to the raw-query path — unlike the legacy parser, which records
  // failures with fallback_used=false.
  const withMeta = (
    error: string,
  ): BenchmarkExpansionRecordVh => ({
    qid,
    query,
    status: "format_error",
    raw_output: rawOutput,
    output: [],
    fallback_used: true,
    error,
    generation: meta,
  });

  if (meta.stop_reason !== "eogToken") {
    return withMeta(
      `generation did not complete naturally (stop_reason=${meta.stop_reason}); response not accepted even if its tail parses`,
    );
  }
  if (parsed.diagnostics[0]?.code === "VH_EMPTY_OUTPUT") {
    return withMeta("model returned empty output");
  }
  if (parsed.diagnostics.length > 0) {
    const diag = parsed.diagnostics[0]!;
    return withMeta(`${diag.code} at line ${diag.line}: ${diag.text}`);
  }
  return {
    qid,
    query,
    status: "ok",
    raw_output: rawOutput,
    output: parsed.output as BenchmarkExpansion[],
    fallback_used: false,
    error: null,
    generation: meta,
  };
}

export type ExpansionGenerationProgress = {
  completed: number;
  total: number;
  qid: string | null;
  elapsed_ms: number;
  last_query_ms: number | null;
  format_errors: number;
  generation_errors: number;
  fallbacks: number;
};

export type ExpansionGenerationSummary = {
  output_path: string;
  schema_version: ExpansionSchemaVersion;
  total: number;
  resumed: number;
  generated: number;
  expansion_passes: number;
  format_errors: number;
  generation_errors: number;
  fallbacks: number;
};

/** Identity inputs recorded at generation start; resume validates against it. */
export type GenerationIdentity = {
  /** Seed passed to the engine for every generation in this artifact. */
  seed: number;
  /** SHA-256 of the full prompt template ({query} placeholder included). */
  prompt_sha256: string;
  /** SHA-256 of the generate-model artifact file. */
  model_sha256: string;
  /** File sha256 of the profile JSON, when a profile drives generation. */
  profile_sha256: string | null;
};

function hashFile(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/**
 * Validate a partial artifact against the current generation identity.
 * Every existing record must carry the same seed and prompt/model/profile
 * hashes — a mismatch means the partial file was produced under a different
 * configuration and must not be silently extended.
 */
function validatePartialIdentity(
  partialText: string,
  identity: GenerationIdentity,
): void {
  if (partialText.trim().length === 0) return;
  for (const [index, line] of partialText.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      continue; // Tail garbage is handled by the record-count recovery path.
    }
    if (typeof value !== "object" || value === null) continue;
    const item = value as Record<string, unknown>;
    if (!("generation" in item)) {
      throw new Error(
        `partial artifact line ${index + 1} has no generation meta — it was not written by this VH profile; refusing to resume`,
      );
    }
    const generation = item.generation as Record<string, unknown> | null;
    if (generation?.prompt_sha256 !== identity.prompt_sha256) {
      throw new Error(
        `partial artifact line ${index + 1} prompt_sha256 mismatch: file has ${JSON.stringify(generation?.prompt_sha256)}, profile pins ${identity.prompt_sha256}`,
      );
    }
    if (generation?.model_sha256 !== identity.model_sha256) {
      throw new Error(
        `partial artifact line ${index + 1} model_sha256 mismatch: file has ${JSON.stringify(generation?.model_sha256)}, profile pins ${identity.model_sha256}`,
      );
    }
    // Absent (pre-profile_sha256) rows cannot prove they share the profile.
    if (generation?.profile_sha256 === undefined || generation.profile_sha256 !== identity.profile_sha256) {
      throw new Error(
        `partial artifact line ${index + 1} profile_sha256 mismatch: file has ${JSON.stringify(generation?.profile_sha256)}, run uses ${JSON.stringify(identity.profile_sha256)}`,
      );
    }
    if (generation.seed !== identity.seed) {
      throw new Error(
        `partial artifact line ${index + 1} seed mismatch: file has ${JSON.stringify(generation.seed)}, run uses ${identity.seed}`,
      );
    }
  }
}

export async function generateExpansionArtifact(options: {
  benchmarkDir: string;
  runName: string;
  /**
   * V2-VH profiled path: strict parsing, generation meta, resume identity.
   * Returns only what the engine observed; `identity` owns seed and hashes.
   */
  generateVh?: (query: string) => Promise<{
    raw_output: string;
    meta: Pick<ExpansionGenerationMeta, "stop_reason" | "generated_tokens" | "elapsed_ms">;
  }>;
  /** Legacy path: plain generation, no meta block, old fallback semantics. */
  generateRaw?: (query: string) => Promise<string>;
  identity?: GenerationIdentity;
  outputPath?: string;
  force?: boolean;
  onProgress?: (progress: ExpansionGenerationProgress) => void;
}): Promise<ExpansionGenerationSummary> {
  const vhMode = options.generateVh !== undefined;
  if (vhMode === (options.generateRaw !== undefined)) {
    throw new Error("provide exactly one of generateVh or generateRaw");
  }
  if (vhMode && !options.identity) {
    throw new Error("generateVh requires identity (seed + prompt/model/profile hashes)");
  }
  const root = resolve(options.benchmarkDir);
  const runName = validateBenchmarkRunName(options.runName);
  const queries = parseQueriesJsonl(
    readFileSync(join(root, "queries.jsonl"), "utf8"),
  );
  const outputPath = resolve(
    options.outputPath ?? join(root, "expansions", `${runName}.jsonl`),
  );
  const partialPath = `${outputPath}.partial`;
  mkdirSync(dirname(outputPath), { recursive: true });

  if (existsSync(outputPath)) {
    if (!options.force) {
      throw new Error(`Expansion artifact already exists: ${outputPath}`);
    }
    rmSync(outputPath);
  }
  if (options.force) rmSync(partialPath, { force: true });
  if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");

  const partialText = readFileSync(partialPath, "utf8");
  const lineCount = partialText.split(/\r?\n/).filter(line => line.length > 0).length;
  if (lineCount > queries.length) {
    throw new Error(`Partial expansion artifact has ${lineCount} rows for ${queries.length} queries`);
  }
  if (vhMode && options.identity) {
    validatePartialIdentity(partialText, options.identity);
  }
  const records: BenchmarkExpansionRecord[] = lineCount === 0
    ? []
    : parseExpansionsJsonlVersioned(partialText, queries.slice(0, lineCount)).records;
  const resumed = records.length;
  const startedAt = performance.now();
  const withIdentity = (
    engine: Pick<ExpansionGenerationMeta, "stop_reason" | "generated_tokens" | "elapsed_ms">,
  ): ExpansionGenerationMeta => ({
    ...engine,
    seed: options.identity!.seed,
    prompt_sha256: options.identity!.prompt_sha256,
    model_sha256: options.identity!.model_sha256,
    profile_sha256: options.identity!.profile_sha256,
  });

  const report = (qid: string | null, lastQueryMs: number | null): void => {
    if (!options.onProgress) return;
    options.onProgress({
      completed: records.length,
      total: queries.length,
      qid,
      elapsed_ms: performance.now() - startedAt,
      last_query_ms: lastQueryMs,
      format_errors: records.filter(record => record.status === "format_error").length,
      generation_errors: records.filter(record => record.status === "generation_error").length,
      fallbacks: records.filter(record => record.fallback_used).length,
    });
  };
  report(null, null);

  for (const query of queries.slice(resumed)) {
    const queryStartedAt = performance.now();
    let record: BenchmarkExpansionRecord;
    try {
      if (vhMode && options.generateVh) {
        const { raw_output, meta } = await options.generateVh(query.query);
        record = parseVhExpansion(query.query, query.qid, raw_output, withIdentity(meta));
      } else if (options.generateRaw) {
        const parsed = parseGeneratedExpansion(query.query, await options.generateRaw(query.query));
        record = { qid: query.qid, query: query.query, ...parsed };
      } else {
        throw new Error("unreachable: no generator");
      }
    } catch (error) {
      record = {
        qid: query.qid,
        query: query.query,
        status: "generation_error",
        raw_output: "",
        output: [],
        // VH policy: the generation failure falls back to the raw-query path.
        fallback_used: vhMode,
        error: error instanceof Error ? error.message : String(error),
      };
      if (vhMode && options.identity) {
        record = {
          ...record,
          generation: withIdentity({
            stop_reason: "error",
            generated_tokens: 0,
            elapsed_ms: Math.round(performance.now() - queryStartedAt),
          }),
        } as BenchmarkExpansionRecordVh;
      }
    }
    records.push(record);
    appendFileSync(partialPath, `${JSON.stringify(record)}\n`, "utf8");
    report(query.qid, performance.now() - queryStartedAt);
  }

  const completedText = readFileSync(partialPath, "utf8");
  const finalParse = parseExpansionsJsonlVersioned(completedText, queries);
  renameSync(partialPath, outputPath);

  const count = (status: ExpansionStatus): number =>
    records.filter(record => record.status === status).length;
  return {
    output_path: outputPath,
    schema_version: finalParse.schema_version,
    total: queries.length,
    resumed,
    generated: queries.length - resumed,
    expansion_passes: count("ok"),
    format_errors: count("format_error"),
    generation_errors: count("generation_error"),
    fallbacks: records.filter(record => record.fallback_used).length,
  };
}

export function fileSha256(path: string): string {
  return hashFile(path);
}
